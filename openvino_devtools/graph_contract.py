# Graph contract experimental code that finds repeated subgraphs in the input graph and contracts them into a single node recursively.
# Copyright (C) 2025 Intel Corporation
# Copyright (C) 2025 Sergey Lyalin

import networkx as nx
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Tuple, Set
import pprint

# Label is a unique identifier of node "kind". Two nodes with the same lable are considered to be matching.
def get_node_label(graph, node):
    return graph.nodes[node].get('label', None)

# Each node can have none, one or multiple output edges. Output edges are grouped by the output port. So there can be multiple edges with the same output port.
def get_edge_out_port(edge_data):
    return edge_data.get('out_port', None)

# Each node can have none, one or multiple input edges. Each input edge has a unique for this node input port. There is no two input edges with the same input port.
def get_edge_in_port(edge_data):
    return edge_data.get('in_port', None)

# Dataclass that represents a signature of a subgraph union of two nodes u and v.
@dataclass(frozen=True)
class SubgraphPairSignature:
    node_u_label: str  # label of a producer part of the pair, it corresponds to one node in the input graph, its label
    node_v_label: str  # label of a consumer part of the pair, it corresponds to one node in the input graph, its label
    inner_edges: Tuple[Tuple[int, int], ...]  # edges that connect u node with v node, each edge is a pair of (u output port, v input port)
    u_input_ports: Tuple[int, ...] # input ports of u node that come from outside of the subgraph
    u_output_ports: Tuple[int, ...] # output ports of u node that goes output of the subgraph
    v_input_ports: Tuple[int, ...] # input ports of v node that come from outside of the subgraph
    v_output_ports: Tuple[int, ...] # output ports of v node that goes output of the subgraph

    def __str__(self):
        return (f"SubgraphSignature(\n"
                f"  node_u_label={self.node_u_label},\n"
                f"  node_v_label={self.node_v_label},\n"
                f"  inner_edges={self.inner_edges},\n"
                f"  u_input_ports={self.u_input_ports},\n"
                f"  u_output_ports={self.u_output_ports},\n"
                f"  v_input_ports={self.v_input_ports},\n"
                f"  v_output_ports={self.v_output_ports}\n"
                f")")

def preprocess_path_counts_dag(graph):
    # Topological sort
    topo_order = list(nx.topological_sort(graph))
    path_counts = {node: {} for node in graph}

    # Initialize path counts
    for node in topo_order:
        path_counts[node][node] = 1  # Each node has one path to itself

    # Compute path counts
    for node in topo_order:
        for neighbor in graph[node]:
            for target, count in path_counts[node].items():
                path_counts[neighbor][target] = path_counts[neighbor].get(target, 0) + count

    return path_counts

def has_alternate_path_dag(graph, path_counts, u, v):
    direct_edge_count = graph.number_of_edges(u, v)
    return path_counts[u].get(v, 0) > direct_edge_count


# use graph type from networkx that allows multiple edges from the same vertices
def find_repeated_pair_subgraphs(graph: nx.MultiDiGraph):
    path_counts_dag = preprocess_path_counts_dag(graph)
    # Dictionary to store frequency of label pairs with ports
    label_pair_with_ports_counts = defaultdict(list)

    # Dictionary to store all pairs of nodes u and v such as there is at least one edge from u to v
    pairs = set()

    # Search for all pairs of nodes u and v such as there is at least one edge from u to v,
    # then for each such pair (u, v) find all the edges that come from u to v.
    for u, v, edge_data in graph.edges(data=True):  # Corrected to unpack three values
        # consider each pair of connected nodes u and v only once
        if (u, v) in pairs:
            continue
        # check if there is an alternative path from u to v
        if has_alternate_path_dag(graph, path_counts_dag, u, v):
            continue
        pairs.add((u, v))

        # traverse all edges that go from u to v
        inner_edges = set()   # identify each edge as a pair of output port, input_port
        # traverse all edges from u to v
        for edge_data in graph.get_edge_data(u, v).values():
            out_port = get_edge_out_port(edge_data)
            in_port = get_edge_in_port(edge_data)
            inner_edges.add((out_port, in_port))
        # print(f"Edges from {u} to {v}: {inner_edges}")

        def collect_input_ports(node, exclude_node=None):
            input_ports = set()
            for src, dst, edge_data in graph.in_edges(node, data=True):
                assert dst == node
                if exclude_node is not None and src == exclude_node:
                    continue
                in_port = get_edge_in_port(edge_data)
                assert in_port not in input_ports  # only one edge per input port
                input_ports.add(in_port)
            return input_ports

        # collect all input edges to u and build a set of unique input port labels
        u_input_ports = collect_input_ports(u)

        # collect all input edges that come to v and not come from u
        v_input_ports = collect_input_ports(v, exclude_node=u)

        def collect_output_ports(node, exclude_node=None):
            output_ports = set()
            for src, dst, edge_data in graph.out_edges(node, data=True):
                assert src == node
                if exclude_node is not None and dst == exclude_node:
                    continue
                out_port = get_edge_out_port(edge_data)
                output_ports.add(out_port)  # there can be multiple edges with the same output port, but we will save only one
            return output_ports

        u_output_ports = collect_output_ports(u, v)
        v_output_ports = collect_output_ports(v)

        # TODO: Some operations are comutative and some are not. We need to take it into account to decrease the number of subgraphs.
        signature = SubgraphPairSignature(
            node_u_label=get_node_label(graph, u),
            node_v_label=get_node_label(graph, v),
            inner_edges=tuple(sorted(inner_edges)),
            u_input_ports=tuple(sorted(u_input_ports)),
            u_output_ports=tuple(sorted(u_output_ports)),
            v_input_ports=tuple(sorted(v_input_ports)),
            v_output_ports=tuple(sorted(v_output_ports))
        )

        label_pair_with_ports_counts[signature].append((u, v))

    return label_pair_with_ports_counts

# Function that takes a list of tuples with pairs. Elements in tuples may intersect or not. The function should return a list of tuples that are not intersect any other tuples in the list using greedy algorithm.
def disjoint_pairs(pairs):
    disjoint_pairs = []
    all_elements = set()
    for pair in pairs:
        # check if s intersects with any of already considered sets
        if not all_elements.intersection(set(pair)):
            disjoint_pairs.append(pair)
            all_elements.update(set(pair))
    return disjoint_pairs

# Function that takes a list of tuples of sets. Returns a list with the same number of elements, where each element is a union of all sets in the corresponding element from the input.
def union_of_tuples(tuples):
    return [set.union(*t) for t in tuples]

# Function that takes a dictionary with values of lists. Each list contains a tuple. Each tuples is a pair. The function calls disjoint_sets for each list.
# It sorts the result by the length of the resulting list keeping the original key from the input dictionary.
def disjoint_ranked_pairs(d):
    disjoint_dict  = {k: disjoint_pairs(v) for k, v in d.items()}
    #print([len(x[1]) for x in disjoint_dict.items()])
    return sorted([(k, v) for k, v in disjoint_dict.items()], key=lambda x: -len(x[1]))

# Function that takes the output of disjoint_ranked_pairs and form a list of not intersected merging pairs
def mergers(d):
    result = OrderedDict()
    visited = set()
    for k, v in d:
        for pair in v:
            if not visited.intersection(set(pair)):
                visited.update(set(pair))
                # print('Adding:', k, v, pair)
                result.setdefault(k, []).append(pair)
    return result

def produce_merge_pairs(graph):
    candidates = find_repeated_pair_subgraphs(graph)
    disjoint = disjoint_ranked_pairs(candidates)
    merge_pairs = mergers(disjoint)
    return merge_pairs

# This function contracts two nodes in the graph into one node.
# It takes 2 arguments: a pair of contracted nodes u->v and subgraph description as a SubgraphPairSignature.
# When creating a new node instead of a given pair u->v, it produces input and output ports marks accoring to the subgraph description as follows:
# 1. Set of input port labels is a range(0, len(u_input_ports + v_input_ports)).
# 2. Set of output port labels is a range(0, len(u_output_ports + v_output_ports)).
# 3. For each edge that goes to u node, there will be an edge to the new contracted node with an input port that corresponds to the index of that input port in the u_input_ports set.
# 4. For each edge that goes to v node from nodes other than u node, there will be a new edge to the new contracted node with an input port that corresponds to the index of that input port of v node in the v_input_ports set + len(u_input_ports).
# 5. All edges that go from u node to v node will be removed.
# 6. For each edge that goes from u node to other nodes (not v), there will be an edge from the new contracted node with an output port that corresponds to the index of that output port in the u_output_ports set.
# 7. For each edge that goes from v node, there will be an edge from the new contracted node with an output port that corresponds to the index of that output port of v node in the v_output_ports set + len(u_output_ports).
# 8. All output port marks for input edges to u and v will kept as in the orginal graph.
# 9. All input port marks for output edges from u and v will kept as in the orginal graph.
def contract_nodes(graph, u, v, signature):
    #print(f"Contracting nodes {u} and {v} with signature {signature}")
    new_node = max(graph.nodes) + 1
    graph.add_node(new_node)
    for src, dst, edge_data in graph.in_edges(u, data=True):
        in_port = get_edge_in_port(edge_data)
        assert in_port in signature.u_input_ports, f"Input port {in_port} is not in {signature.u_input_ports}"
        graph.add_edge(src, new_node, in_port=signature.u_input_ports.index(in_port), out_port=get_edge_out_port(edge_data))
    for src, dst, edge_data in graph.in_edges(v, data=True):
        if src == u:
            continue
        in_port = get_edge_in_port(edge_data)
        graph.add_edge(src, new_node, in_port=signature.v_input_ports.index(in_port) + len(signature.u_input_ports), out_port=get_edge_out_port(edge_data))
    for src, dst, edge_data in graph.out_edges(u, data=True):
        if dst == v:
            continue
        out_port = get_edge_out_port(edge_data)
        assert out_port in signature.u_output_ports, f"Output port {out_port} is not in {signature.u_output_ports}"
        graph.add_edge(new_node, dst, out_port=signature.u_output_ports.index(out_port), in_port=get_edge_in_port(edge_data))
    for src, dst, edge_data in graph.out_edges(v, data=True):
        out_port = get_edge_out_port(edge_data)
        graph.add_edge(new_node, dst, out_port=signature.v_output_ports.index(out_port) + len(signature.u_output_ports), in_port=get_edge_in_port(edge_data))
    graph.remove_node(u)
    graph.remove_node(v)
    return new_node

# function that takes a dictionary with keys of type SubgraphPairSignature and values of type list of sets. Each set contains nodes that form a subgraph with the corresponding signature.
# The function should modify the graph in-place by contradicting all nodes in each set with a new node that has the lable from the second parameter of the function, which is a dictionary with keys of type SubgraphPairSignature and values of type string.
def contract_subgraphs(graph, merge_pairs, new_labels):
    for signature, pairs in merge_pairs.items():
        for pair in pairs:
            u, v = pair
            new_node = contract_nodes(graph, u, v, signature)
            graph.nodes[new_node]['label'] = new_labels[signature]

# Function that prints networkx graph with pseudographics in console.
def print_graph(graph):
    for node in graph.nodes(data=True):
        print(f"Node {node[0]}: {node[1]}")
    for u, v, edge_data in graph.edges(data=True):
        print(f"Edge from {u} to {v}: {edge_data}")

# Prints a graph as a program where nodes are operations and edges are data flows.
# Each statement has a form: tI = statement(in_port_0=tX, in_port_1=tY, ...), where each Z in each port_Z contains labels from edges which corresponds to in_port mark. So if an edge has attribute in_port=0, then it will be in_port_0.
# Statements are printed in topological order.
def print_dag_as_program(graph: nx.MultiDiGraph, name):
    topo_order = list(nx.topological_sort(graph))
    output_port_marks = {}  # dict that maps an edge identified as source node and output port lable from the edge data to the variable name

    # Handle all __Input__ nodes first
    for node in topo_order:
        if get_node_label(graph, node) == "__Input__":
            index = graph.nodes[node]['index']
            output_port_marks[(node, 0)] = f"t{index}"

    args = ",".join(f"t{i}" for i in range(len(output_port_marks)))
    print(f'def {name}({args}):')

    for node in topo_order:
        label = get_node_label(graph, node)
        if label in ["__Input__", "__Output__"]:
            continue
        in_edges = graph.in_edges(node, data=True)
        if in_edges:
            max_input_ports = max([get_edge_in_port(edge_data) for _, _, edge_data in in_edges]) + 1
            input_args = [None] * max_input_ports
            for src, dst, edge_data in in_edges:
                in_port = get_edge_in_port(edge_data)
                value = output_port_marks[(src, get_edge_out_port(edge_data))]
                input_args[in_port] = value
            input_args = [arg if arg is not None else "None" for arg in input_args]
        else:
            input_args = []
        output_values = {}
        visited = set()
        for src, dst, edge_data in graph.out_edges(node, data=True):
            out_port = get_edge_out_port(edge_data)
            if out_port in visited:
                continue
            visited.add(out_port)
            value = f"t{len(output_port_marks) + len(output_values)}"
            output_values[out_port] = value
        # fill the gaps in the output_values, part of them won't be used in the generated program
        if output_values:
            for i in range(max(output_values.keys())+1):
                if i not in output_values:
                    output_values[i] = f"t{len(output_port_marks) + len(output_values)}"
        output_port_marks.update({(node, k): v for k, v in output_values.items()})
        # convert output_values from dict to list where key is an index
        output_values = [v for k, v in sorted(output_values.items())]
        output_part = ", ".join(output_values)
        input_part = ", ".join(input_args)
        print(4*' ', end="")
        if output_part:
            print(output_part, end=" = ")
        print(f"{label}({input_part})")

    # Handle all __Output__ nodes by collecting all of them and then printing a single return tX, tY, ... statement in order of `index` attribute attached to each __Output__ node
    output_nodes = [node for node in topo_order if get_node_label(graph, node) == "__Output__"]
    output_nodes.sort(key=lambda node: graph.nodes[node]['index'])
    output_values = []
    for node in output_nodes:
        for src, dst, data in graph.in_edges(node, data=True):
            output_values.append(output_port_marks[(src, get_edge_out_port(data))])

    output_part = ", ".join(output_values)
    if output_part:
        print(4*' ', end="")
        print(f"return {output_part}")




# Represent a SubgraphPairSignature as an nx.MultiDiGraph instance.
# The core of this graph is two nodes from the subgraph: producer (u) and consumer (v). The labels on those nodes correspond to u and v labels from the signature.
# Those nodes are connected with edges from the inner_edges field of the signature. Each edge is a pair of output port of u and input port of v.
# There are also auxiliary nodes that represent external input and output ports.
# Each input auxiliary node has a label "__Input__" and additional attribute `index`=i where i is an index of the input port in the u_input_ports + v_input_ports set. The numbering matches one that contract_nodes function uses when numbering input ports.
# Each output auxiliary node has a label "__Output__" and additional attribute `index` where i is an index of the output port in the u_output_ports + v_output_ports set. The numbering matches one that contract_nodes function uses when numbering output ports.
def subgraph_signature_to_graph(signature: SubgraphPairSignature):
    graph = nx.MultiDiGraph()
    graph.add_node(0, label=signature.node_u_label)
    graph.add_node(1, label=signature.node_v_label)
    node_index_base = 2
    input_index_base = 0
    output_index_base = 0
    for u_out_port, v_in_port in signature.inner_edges:
        graph.add_edge(0, 1, out_port=u_out_port, in_port=v_in_port)
    for u_input_port in signature.u_input_ports:
        graph.add_node(node_index_base, label="__Input__", index=input_index_base)
        graph.add_edge(node_index_base, 0, in_port=u_input_port, out_port=0)
        node_index_base += 1
        input_index_base += 1
    for v_input_port in signature.v_input_ports:
        graph.add_node(node_index_base, label="__Input__", index=input_index_base)
        graph.add_edge(node_index_base, 1, in_port=v_input_port, out_port=0)
        node_index_base += 1
        input_index_base += 1
    for u_output_port in signature.u_output_ports:
        graph.add_node(node_index_base, label="__Output__", index=output_index_base)
        graph.add_edge(1, node_index_base, in_port=0, out_port=u_output_port)
        node_index_base += 1
        output_index_base += 1
    for v_output_port in signature.v_output_ports:
        graph.add_node(node_index_base, label="__Output__", index=output_index_base)
        graph.add_edge(1, node_index_base, in_port=0, out_port=v_output_port)
        node_index_base += 1
        output_index_base += 1
    assert node_index_base == len(graph)
    assert input_index_base == len(signature.u_input_ports) + len(signature.v_input_ports)
    assert output_index_base == len(signature.u_output_ports) + len(signature.v_output_ports)
    return graph

def test_find_repeated_pair_subgraphs():
    graph = nx.MultiDiGraph()
    n = 10
    # create a few nodes and connect all of them with edges according to the description above
    for i in range(6+n):
        graph.add_node(i, label="A" if i%2==0 else "B")

    graph.add_edge(0, 1, out_port=0, in_port=0)
    graph.add_edge(0, 2, out_port=0, in_port=3)
    graph.add_edge(1, 3, out_port=0, in_port=0)
    graph.add_edge(2, 3, out_port=0, in_port=1)
    graph.add_edge(3, 4, out_port=0, in_port=0)
    graph.add_edge(4, 5, out_port=0, in_port=0)
    for i in range(n):
        graph.add_edge(5+i, 6+i, out_port=0, in_port=0)

    graph.add_node(100, label="C")

    graph.add_edge(7, 100, out_port=0, in_port=0)
    graph.add_edge(100, 8, out_port=0, in_port=1)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(find_repeated_pair_subgraphs(graph))

    print('-'*80)

    subgraph_base_index = 0

    non_terminal_labels = {}  # dict that maps lable to subgraph description

    while(True):
        print_dag_as_program(graph, 'main')
        print('\n--------------------------------------------------\n')
        merge_pairs = produce_merge_pairs(graph)
        if not merge_pairs:
            break
        # create new labels in form "S" + str(i) for each signature, build a dict with subgraph description as a key and new label as a value
        # labels can have any type including subgraph description by it is more convenient to use strings for debugging
        new_labels = {k: f"S{i+subgraph_base_index}" for i, (k, _) in enumerate(merge_pairs.items())}
        non_terminal_labels.update({v: k for k, v in new_labels.items()})
        for new_label, signature in non_terminal_labels.items():
            subgraph_graph = subgraph_signature_to_graph(signature)
            print_dag_as_program(subgraph_graph, new_label)
            print()
        subgraph_base_index += len(new_labels)
        contract_subgraphs(graph, merge_pairs, new_labels)
        #print_graph(graph)

    # print all non-terminals with their labels
    #print("Non-terminals:")
    #pp.pprint(non_terminal_labels)

if __name__ == '__main__':
    test_find_repeated_pair_subgraphs()
