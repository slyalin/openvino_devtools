from openvino.runtime.utils.node_factory import NodeFactory
from openvino.runtime import Node, Output, Model, PartialShape
from typing import Optional, List, Dict, Any, Union
import numpy as np


def get_attribute_name_in_python(a):
    return "'" + a + "'"

def get_attribute_value_in_python(a):
    return "'" + a + "'" if isinstance(a, str) else "'" + str(a) + "'" if isinstance(a, PartialShape) else a

def get_output_names(op):
    output_names = '[' + ', '.join('{' + ', '.join(sorted(f"'{name}'" for name in port.get_names())) + '}' for port in op.outputs()) + ']'  #FIXME: sorting names as a workaround for non deterministic order
    if any(port.get_names() for port in op.outputs()):
        return ', output_names=' + str(output_names)
    else:
        return ''

def align_text(p1, p2, width=60):
    p1 += ' '*max(width - len(p1), 0)
    return p1 + p2

def get_sink_index(model, op):
    return [id(s) for s in model.get_sinks()].index(id(op))

def sort_dict(d):
    return [d[i] for i in range(len(d))]


class OpFactory:
    def __init__(self, opset):
        self.factory = NodeFactory(opset)

    def _set_names(self, node, node_name, output_names):
        if node_name is not None:
            node.set_friendly_name(node_name)
        if output_names is not None:
            assert(node.get_output_size() == len(output_names))
            for i in range(node.get_output_size()):
                node.output(i).get_tensor().set_names(output_names[i])
        return node

    def Constant(self, arguments, attributes, node_name=None, output_names=None):
        if isinstance(arguments, Model):
            assert isinstance(attributes, int)
            return arguments.get_ordered_ops()[attributes].output(0)

    def Parameter(self, attributes, node_name=None, output_names=None):
        if 'shape' in attributes:
            attributes['shape'] = PartialShape(attributes['shape'])
        node = self.factory.create('Parameter', [], attributes)
        return self._set_names(node, node_name, output_names).output(0)

    def ReadValue(self, arguments, attributes, node_name=None, output_names=None):
        if 'variable_shape' in attributes:
            attributes['variable_shape'] = PartialShape(attributes['variable_shape'])
        node = self.factory.create('ReadValue', arguments, attributes)
        return  self._set_names(node, node_name, output_names).output(0)

    def __getattr__(self, optype):
        def creator(
            arguments: Optional[List[Union[Node, Output]]] = None,
            attributes: Optional[Dict[str, Any]] = None,
            node_name: str = None,
            output_names: str = None
        ):
            node = self.factory.create(optype, arguments, attributes)
            outputs =  self._set_names(node, node_name, output_names).outputs()
            if len(outputs) == 1:
                return outputs[0]
            else:
                return tuple(outputs)
        return creator

def outputs_to_nodes(outputs):
    return [output.get_node() for output in outputs]

class Operation:
    def __init__(self, printer, model, i, op):
        self.model = model
        self.i = i
        self.op = op
        self.relative_id_cache = None
        self.relative_id_wo_const_cache = None
        self.printer = printer
        # register all inputs and outputs ids
        # FIXME: avoid calling this function twice (here and in that place where the entire program is constructed), it is too bold
        self.get_python_code()

    def get_python_code(self, with_node_names=False, with_tensor_names=True):
        return self.printer.get_op_statement(self.model, self.i, self.op, with_node_names=with_node_names, with_tensor_names=with_tensor_names)

    def relative_id(self):
        if self.relative_id_cache is None:
            self.relative_id_cache = [self.op.get_type_name(), [get_tensor_distance(self.op.output(0), port.get_source_output()) for port in self.op.inputs()]]
        return self.relative_id_cache

    def relative_id_wo_const(self):
        if self.relative_id_wo_const_cache is None:
            self.relative_id_wo_const_cache = [self.op.get_type_name(), [get_tensor_distance(self.op.output(0), port.get_source_output()) if port.get_source_output().get_node().get_type_name() != 'Constant' else 0 for port in self.op.inputs()]]
        return self.relative_id_wo_const_cache


class ModelPrinter:

    def __init__(self):
        self.tensor_counter = 0
        self.tensor_dict = {}
        self.parameters = {}
        self.results = {}
        self.sinks = {}

    def get_tensor(self, tensor):
        if tensor in self.tensor_dict:
            result = self.tensor_dict[tensor]
        else:
            result = self.tensor_dict[tensor] = f't{self.tensor_counter}'
            self.tensor_counter += 1
        return result

    def get_tensor_distance(self, tensor1, tensor2):
        return int(self.tensor_dict[tensor1][1:]) - int(self.tensor_dict[tensor2][1:])

    def get_op_statement(self, model, i, op, with_node_names=False, with_tensor_names=True):
        inputs = '[' + ', '.join([self.get_tensor(port.get_source_output()) for port in op.inputs()]) + '], '
        outputs = ', '.join([self.get_tensor(port) for port in op.outputs()])
        attrs = ', '.join([f'{get_attribute_name_in_python(k)}: {get_attribute_value_in_python(v)}' for k, v in op.get_attributes().items()])
        input_types = ', '.join([port.get_element_type().get_type_name() + str(port.get_partial_shape()) for port in op.inputs()])
        output_types = ', '.join([port.get_element_type().get_type_name() + str(port.get_partial_shape()) for port in op.outputs()])
        if with_node_names:
            node_name = f', node_name="{op.get_friendly_name()}"'
        else:
            node_name = ''
        if op.get_type_name() == 'Constant':
            if op.get_data().size <= 10:
                value = '(' + str(op.get_data().tolist()) + ')'
            else:
                value = ''
            return align_text(f'{outputs} = opset.{op.get_type_name()}(model, {i}{node_name}{get_output_names(op)})  ', f'# {input_types} -> {output_types}{value}')
        elif op.get_type_name() == 'Parameter':
            self.parameters[model.get_parameter_index(op)] = self.get_tensor(op.output(0))
            inputs = ''
        elif op.get_type_name() == 'Result':
            self.results[model.get_result_index(op.output(0))] = self.get_tensor(op.output(0))
        elif op.get_type_name() == 'Assign':
            self.sinks[get_sink_index(model, op)] = self.get_tensor(op.output(0))
        attrs = '{' + attrs + '}'
        if with_tensor_names:
            output_names = get_output_names(op)
        else:
            output_names = ''
        return align_text(f'{outputs} = opset.{op.get_type_name()}({inputs}{attrs}{node_name}{output_names})  ', f'# {input_types} -> {output_types}')

    def implement_model(self, model, name='build_model', with_node_names=False, path_to_source_model=None, entry_point=False, with_tensor_names=True):
        new_op_trace = []
        result = f'def {name}(model):\n'
        indent = '    '
        result += indent + 'import openvino as ov\n'
        result += indent + 'from openvino_devtools.builder import OpFactory, outputs_to_nodes\n'
        result += indent + 'opset = OpFactory(\'opset13\')\n'
        ordered_ops = model.get_ordered_ops()
        # Accumulate parameters, results and sinks, then sort them in the same order they appear in model parameters and sinks vectors
        # get_ordered_ops doesn't respect that order for some unknown reason
        postponed_parameters = []
        postponed_assigns = []
        postponed_results = []
        for i in range(len(ordered_ops)):
            op = ordered_ops[i]
            if op.get_type_name() == 'Constant':
                pass
            if op.get_type_name() == 'Parameter':
                postponed_parameters.append(i)
            elif op.get_type_name() == 'Result':
                postponed_results.append(i)
            elif op.get_type_name() == 'Assign':
                postponed_assigns.append(i)
            else:
                if len(postponed_parameters):
                    # flush all parameters at once in the correct order
                    assert len(postponed_parameters) == len(model.get_parameters())
                    postponed_parameters.sort(key=lambda i: model.get_parameter_index(ordered_ops[i]))
                    for j in postponed_parameters:
                        new_op_trace.append(Operation(self, model, i, ordered_ops[j]))
                    postponed_parameters = []
                new_op_trace.append(Operation(self, model, i, op))

        # TODO: The assert below started firing when other types of Sinks except Assign were introduced in the model. Investigate why we need this assert.
        # assert len(postponed_assigns) == len(model.get_sinks())

        def sort_and_flush(result, node_indices, order_func):
            if len(node_indices):
                # flush all assigns at once in the correct order
                for j in node_indices:
                    new_op_trace.append(Operation(self, model, i, ordered_ops[j]))

        sort_and_flush(result, postponed_assigns, lambda op: get_sink_index(model, op))
        sort_and_flush(result, postponed_results, lambda op: model.get_result_index(op.output(0)))

        result += '\n'

        for op in new_op_trace:
            result += indent + op.get_python_code(with_node_names=with_node_names, with_tensor_names=with_tensor_names) + '\n'

        result += '\n'
        result += indent + 'parameters = [' + ', '.join(sort_dict(self.parameters)) + ']\n'
        result += indent + 'results = [' + ', '.join(sort_dict(self.results)) + ']\n'
        result += indent + 'sinks = [' + ', '.join(sort_dict(self.sinks)) + ']\n'
        result += indent + 'return ov.Model(outputs_to_nodes(results), outputs_to_nodes(sinks), outputs_to_nodes(parameters), name=model.get_friendly_name())'
        if entry_point:
            result += '\n\n\nif __name__ == \'__main__\':\n'
            result += indent + 'from openvino_devtools.builder import regenerate_model\n'
            path_to_source_model = 'None' if path_to_source_model is None else f"'{path_to_source_model}'"
            result += indent + f'regenerate_model(build_model, path_to_source_model={path_to_source_model})'
        return result


class CppModelPrinter(ModelPrinter):
    def get_op_statement(self, model, i, op, with_node_names=False, with_tensor_names=True):
        temp_var = None
        inputs = '{' + ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs()) + '}'
        outputs = ', '.join([self.get_tensor(port) for port in op.outputs()])
        attrs = ', '.join([f'{get_attribute_name_in_python(k)}: {get_attribute_value_in_python(v)}' for k, v in op.get_attributes().items()])
        input_types = ', '.join([port.get_element_type().get_type_name() + str(port.get_partial_shape()) for port in op.inputs()])
        output_types = ', '.join([port.get_element_type().get_type_name() + str(port.get_partial_shape()) for port in op.outputs()])
        if with_node_names:
            node_name = f', node_name="{op.get_friendly_name()}"'
        else:
            node_name = ''
        element_type = op.get_element_type().get_type_name()
        try:
            shape = str(op.get_shape()).replace('[', '{').replace(']', '}')
            if op.get_type_name() == 'Constant':
                shape = 'Shape' + shape
            else:
                shape = 'PartialShape' + shape
        except AttributeError:
            shape = None
        ov_type_to_cpp = {
            'i32': 'int32_t',
            'i64': 'int64_t',
            'f32': 'float',
        }
        if op.get_type_name() == 'Constant':
            if op.get_data().size <= 10:
                value = '(' + str(op.get_data().tolist()) + ')'
            else:
                value = ''
            old = align_text(f'{outputs} = opset.{op.get_type_name()}(model, {i}{node_name}{get_output_names(op)})  ', f'# {input_types} -> {output_types}{value}')
            values = op.get_data().tolist()
            fill_value = None
            if isinstance(values, (int, float)):
                str_values = '{' + str(values) + '}'
                n_dims = 1
            else:
                n_dims = len(op.get_data().shape)
                str_values = str(op.get_data().tolist()).replace('[', '{').replace(']', '}')
                vector_type = 'vector<' * n_dims + ov_type_to_cpp[element_type] + '>' * n_dims
                values = f'{vector_type}{str_values}'
                fill_value = op.get_data().flatten()[0]
                if (fill_value == op.get_data()).all():
                    if op.get_data().dtype == np.float32:
                        suffix = 'f'
                    else:
                        suffix = ''
                    values = str(fill_value) + suffix
            new = align_text(f'auto {outputs} = make_shared<{op.get_type_name()}>({element_type}, {shape}, {str(values).lower()});  ', f'// {input_types} -> {output_types}{value}')
            return new
        elif op.get_type_name() == 'Parameter':
            self.parameters[model.get_parameter_index(op)] = self.get_tensor(op.output(0))
            inputs = None
        elif op.get_type_name() == 'Result':
            self.results[model.get_result_index(op.output(0))] = self.get_tensor(op.output(0))
        elif op.get_type_name() == 'Assign':
            self.sinks[get_sink_index(model, op)] = self.get_tensor(op.output(0))
        attrs = '{' + attrs + '}'
        if with_tensor_names:
            output_names = get_output_names(op)
        else:
            output_names = ''
        if op.get_type_name() == 'Result':
            inputs = self.get_tensor(op.inputs()[0].get_source_output())
            element_type = None
        elif op.get_type_name() == 'ShapeOf':
            inputs = self.get_tensor(op.inputs()[0].get_source_output())
            element_type = None
        elif op.get_type_name() == 'Gather':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'Reshape':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs()) + ', ' + str(op.get_special_zero()).lower()
            element_type = None
        elif op.get_type_name() == 'Convert':
            inputs = self.get_tensor(op.inputs()[0].get_source_output())
        elif op.get_type_name() == 'Power':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs()) + ', "numpy"'
            element_type = None
        elif op.get_type_name() == 'Unsqueeze':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'Concat':
            inputs = 'NodeVector{' + ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs()) + '}, ' + str(op.get_axis())
            element_type = None
        elif op.get_type_name() == 'Divide':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs()) + ', "numpy"'
            element_type = None
        elif op.get_type_name() == 'Floor':
            inputs = self.get_tensor(op.inputs()[0].get_source_output())
            element_type = None
        elif op.get_type_name() == 'Transpose':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'Multiply':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs()) + ', "numpy"'
            element_type = None
        elif op.get_type_name() == 'Ceiling':
            inputs = self.get_tensor(op.inputs()[0].get_source_output())
            element_type = None
        elif op.get_type_name() == 'Broadcast':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'ReduceProd':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'Range':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
        elif op.get_type_name() == 'Less':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'GreaterEqual':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'Subtract':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'Add':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'Slice':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'ScatterNDUpdate':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'ReduceSum':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'Interpolate':
            attributes = op.get_attributes()
            pads_begin = f'vector<size_t>{str(attributes['pads_begin']).replace('[', '{').replace(']', '}')}'
            pads_end = f'vector<size_t>{str(attributes['pads_end']).replace('[', '{').replace(']', '}')}'
            temp_var = f'Interpolate::InterpolateAttrs {outputs}_attrs{{Interpolate::InterpolateMode::{attributes['mode'].upper()}, Interpolate::ShapeCalcMode::{attributes['shape_calculation_mode'].upper()}, {pads_begin}, {pads_end}, Interpolate::CoordinateTransformMode::{attributes['coordinate_transformation_mode'].upper()}, Interpolate::NearestMode::{attributes['nearest_mode'].upper()}, {str(attributes['antialias']).lower()}, {attributes['cube_coeff']}}};'
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            inputs += f', {outputs}_attrs'
            element_type = None
        elif op.get_type_name() == 'Tile':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs())
            element_type = None
        elif op.get_type_name() == 'ReduceMin':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs()) + ', ' + str(op.get_keep_dims()).lower()
            element_type = None
        elif op.get_type_name() == 'Bucketize':
            inputs = ', '.join(self.get_tensor(port.get_source_output()) for port in op.inputs()) + ', ' + element_type + ', ' + str(op.get_with_right_bound()).lower()
            element_type = None
        elif op.get_type_name() == 'Loop':
            print('recursion with op.set_function()')
        old = align_text(f'{outputs} = opset.{op.get_type_name()}({inputs}{attrs}{node_name}{output_names})  ', f'# {input_types} -> {output_types}')
        # TODO: typed var names
        # TODO: op.visit_attributes(AttributeVisitor)
        # print(op.get_type_name())
        args = ', '.join(str(arg) for arg in (inputs, element_type, shape) if arg is not None)
        new = align_text(f'auto {outputs} = make_shared<{op.get_type_name()}>({args});  ', f'// {input_types} -> {output_types}')
        output_names = ', '.join(', '.join(sorted(f'"{name}"' for name in port.get_names())) for port in op.outputs())
        indent = '    '
        if op.get_type_name() == 'Parameter' or op.get_type_name() == 'Result':
            new += f'\n{indent}{outputs}->output(0).get_tensor().set_names({{{output_names}}});'
        if temp_var is not None:
            new = temp_var + '\n' + indent + new
        return new
        # py ./phi4-mm_test.py && PYTHONPATH=./ py ./openvino_devtools/ov2cpp.py ref/a.xml > build-model.cpp && g++ -O3 -DNDEBUG -Wall -Wextra -pedantic -Werror --std c++23 -Xlinker -rpath -Xlinker /home/vzlobin/z/v/bin/intel64/Debug -lpthread -isystem /home/vzlobin/z/v/src/core/include/ -isystem /home/vzlobin/z/v/src/inference/include/ -isystem /home/vzlobin/z/repr/benchmark/include/ build-model.cpp /home/vzlobin/z/v/bin/intel64/Debug/libopenvino.so.2025.2.0 && ./a.out && PYTHONPATH=./ py ./openvino_devtools/ovdiff.py ref res

    def implement_model(self, model, with_node_names=False, path_to_source_model=None, entry_point=False, with_tensor_names=True):
        new_op_trace = []
        result = f'#include <openvino/openvino.hpp>\n\n#include "openvino/opsets/opset13.hpp"\n\nstatic std::shared_ptr<ov::Model> build_model() {{\n'
        indent = '    '
        result += indent + 'using namespace ov;\n'
        result += indent + 'using namespace element;\n'
        result += indent + 'using namespace opset13;\n'
        result += indent + 'using namespace std;\n'
        ordered_ops = model.get_ordered_ops()
        # Accumulate parameters, results and sinks, then sort them in the same order they appear in model parameters and sinks vectors
        # get_ordered_ops doesn't respect that order for some unknown reason
        postponed_parameters = []
        postponed_assigns = []
        postponed_results = []
        for i in range(len(ordered_ops)):
            op = ordered_ops[i]
            if op.get_type_name() == 'Constant':
                pass
            if op.get_type_name() == 'Parameter':
                postponed_parameters.append(i)
            elif op.get_type_name() == 'Result':
                postponed_results.append(i)
            elif op.get_type_name() == 'Assign':
                postponed_assigns.append(i)
            else:
                if len(postponed_parameters):
                    # flush all parameters at once in the correct order
                    assert len(postponed_parameters) == len(model.get_parameters())
                    postponed_parameters.sort(key=lambda i: model.get_parameter_index(ordered_ops[i]))
                    for j in postponed_parameters:
                        new_op_trace.append(Operation(self, model, i, ordered_ops[j]))
                    postponed_parameters = []
                new_op_trace.append(Operation(self, model, i, op))

        # TODO: The assert below started firing when other types of Sinks except Assign were introduced in the model. Investigate why we need this assert.
        # assert len(postponed_assigns) == len(model.get_sinks())

        def sort_and_flush(result, node_indices, order_func):
            if len(node_indices):
                # flush all assigns at once in the correct order
                for j in node_indices:
                    new_op_trace.append(Operation(self, model, i, ordered_ops[j]))

        sort_and_flush(result, postponed_assigns, lambda op: get_sink_index(model, op))
        sort_and_flush(result, postponed_results, lambda op: model.get_result_index(op.output(0)))

        result += '\n'

        for op in new_op_trace:
            result += indent + op.get_python_code(with_node_names=with_node_names, with_tensor_names=with_tensor_names) + '\n'

        result += '\n'
        result += indent + f'ResultVector results{{{', '.join(sort_dict(self.results))}}};\n'
        result += indent + f'SinkVector sinks{{{', '.join(sort_dict(self.sinks))}}};\n'
        result += indent + f'ParameterVector parameters{{{', '.join(sort_dict(self.parameters))}}};\n'
        result += indent + f'return make_shared<Model>(results, sinks, parameters, "{model.get_friendly_name()}");\n'
        result += '}'
        if entry_point:
            result += """

int main() {
    auto model = build_model();
    auto ireq = ov::Core{}.compile_model(model, "CPU").create_infer_request();
    // std::cout << ireq.get_tensor("1").get_shape() << '\\n';
    bool compress_to_fp16 = false;
    ov::save_model(model, "res_separators/a.xml", compress_to_fp16);
}"""
        return result


def regenerate_model(model_builder, path_to_source_model=None):
    import openvino as ov
    core = ov.Core()
    from sys import argv

    def usage():
        print('Usage:')
        print(f'    {argv[0]} [<PATH TO SOURCE OV IR XML>] <PATH TO DESTINATION OV IR XML>')
        print(f'Optional <PATH TO SOURCE OV IR XML> is used to restore all Constants content. If not provided the original path "{path_to_source_model}" is used. If the path was not provided when script was generated then this argument is mandatory.')

    if len(argv) < 2:
        print('ERROR: Too few command line arguments.')
        usage()
        return
    elif len(argv) == 2:
        if(path_to_source_model is None):
            print('ERROR: No <PATH TO SOURCE OV IR XML> is provided and there was no path provided when script was generated.')
            usage()
            return
        path_to_generated_model = argv[1]
    elif len(argv) == 3:
        path_to_source_model = argv[1]
        path_to_generated_model = argv[2]
    else:
        print('ERROR: Too many command line arguments')
        usage()
        return
    source_model = core.read_model(path_to_source_model)
    generated_model = model_builder(source_model)
    ov.save_model(generated_model, path_to_generated_model, compress_to_fp16=False)