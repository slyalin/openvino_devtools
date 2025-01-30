import openvino as ov
import networkx as nx
from graph_contract import print_dag_as_program, contract_recursive

def openvino_model_to_nx_graph(model: ov.Model) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    ordered_ops = model.get_ordered_ops()

    # Add nodes with labels
    for i, op in enumerate(ordered_ops):
        op_type = op.get_type_name()
        attributes = {k: v for k, v in op.get_attributes().items()}
        label = f"{op_type}({attributes})"
        graph.add_node(i, label=label)

    # Add edges with in_port and out_port marks
    for i, op in enumerate(ordered_ops):
        for j, input in enumerate(op.inputs()):
            source_op = ordered_ops.index(input.get_source_output().get_node())
            source_port = input.get_source_output().get_index()
            target_port = j
            graph.add_edge(source_op, i, out_port=source_port, in_port=target_port)

    assert nx.is_directed_acyclic_graph(graph)
    nx.topological_sort(graph)

    return graph

# Example usage
if __name__ == "__main__":
    core = ov.Core()
    import sys
    model = core.read_model(sys.argv[1])
    nx_graph = openvino_model_to_nx_graph(model)
    contract_recursive(nx_graph)
