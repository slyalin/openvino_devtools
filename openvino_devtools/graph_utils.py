from dataclasses import dataclass
import openvino as ov
import networkx as nx
from graph_contract import print_dag_as_program, contract_recursive

@dataclass(frozen=True)
class Label:
    op_type: str
    attributes: str
    variation: str  # to differentiate nodes with the same op_type and attributes: Constant, Parameter, Result

def print_label(label, args, specials) -> str:
    args = ",".join(args)
    if isinstance(label, Label):
        if label.op_type in specials:
            return specials[label.op_type].str(label)
        return f"{label.op_type}({args}" + (f", {label.attributes})" if label.attributes else ")")
    elif isinstance(label, str):
        return f"{label}({args})"
    assert False, "Unknown label type " + type(label)


def op_to_label(op: ov.Node, specials) -> Label:
    # TODO: take an opset as well
    op_type = op.get_type_name()
    attributes = {k: v for k, v in op.get_attributes().items()}
    if op_type in specials:
        variation = specials[op_type].new(op)
    else:
        variation = 0
    return Label(op_type, ",".join(f"{k}='{v}'" for k, v in attributes.items()), variation)

class Counter:
    def __init__(self):
        self.count = 0

    def new(self, op: ov.Node):
        self.count += 1
        return str(self.count - 1)

    def str(self, label):
        return f"{label.op_type}({label.variation}" + (f", {label.attributes})" if label.attributes else ")")

class ConstantCounter:
    def __init__(self, constants, threshold=10):
        self.threshold = threshold
        self.count = 0  # count constant bigger than `threshold` elements
        self.constants = constants  # store the constants to be able to refer to them later

    def new(self, op: ov.Node):
        data = op.get_data()
        if data.size <= self.threshold:     # constant with larger number of elements are treated as all different, content is not checked
            port = op.output(0)
            variant = f'{port.get_element_type().get_type_name()}{port.get_partial_shape()}({str(op.get_data().tolist())})'
        else:
            variant = str(self.count)
            self.count += 1
        self.constants[variant] = op
        return variant

    def str(self, label):
        return f"{label.op_type}({label.variation})"

def openvino_model_to_nx_graph(model: ov.Model, specials) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    ordered_ops = model.get_ordered_ops()

    # Add nodes with labels
    for i, op in enumerate(ordered_ops):
        label = op_to_label(op, specials)
        graph.add_node(i, label=label)

    # Add edges with in_port and out_port marks
    for i, op in enumerate(ordered_ops):
        for j, input in enumerate(op.inputs()):
            source_op = ordered_ops.index(input.get_source_output().get_node())
            source_port = input.get_source_output().get_index()
            target_port = j
            graph.add_edge(source_op, i, out_port=source_port, in_port=target_port)

    assert nx.is_directed_acyclic_graph(graph)
    return graph

# Example usage
if __name__ == "__main__":
    core = ov.Core()
    import sys
    model = core.read_model(sys.argv[1])

    constants = {}

    specials = {
        'Constant': ConstantCounter(constants),
        'Parameter': Counter(),
        'Result': Counter()
    }

    nx_graph = openvino_model_to_nx_graph(model, specials)
    contract_recursive(nx_graph, lambda lable, args: print_label(lable, args, specials))
