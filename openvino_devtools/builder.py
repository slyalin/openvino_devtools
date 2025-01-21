from openvino.runtime.utils.node_factory import NodeFactory
from openvino.runtime import Node, Output, Model, PartialShape
from typing import Optional, List, Dict, Any, Union


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
        return self.printer.get_op_statement_in_python(self.model, self.i, self.op, with_node_names=with_node_names, with_tensor_names=with_tensor_names)

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

    def get_op_statement_in_python (self, model, i, op, with_node_names=False, with_tensor_names=True):
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

    def model_to_python(self, model, name='build_model', with_node_names=False, path_to_source_model=None, entry_point=False, with_tensor_names=True):
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

        assert len(postponed_assigns) == len(model.get_sinks())

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