import pickle
import networkx as nx
import argparse


def load_graph(file_path):
    with open(file_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def find_nodes_with_multiple_parents(graph):
    """Find nodes with more than one parent in a directed graph."""
    parent_counts = {node: 0 for node in graph.nodes()}
    for _, child in graph.edges():
        parent_counts[child] += 1
    return [node for node, count in parent_counts.items() if count > 1]


def process_graph(input_file):
    graph = load_graph(input_file)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # Check if the graph is a tree
    if nx.is_tree(graph):
        print("Graph is a tree")
    else:
        print("Graph is not a tree")
        if nx.is_directed(graph):
            nodes_with_multiple_parents = find_nodes_with_multiple_parents(graph)
            if nodes_with_multiple_parents:
                print(
                    "Nodes with multiple parents (violating tree structure):",
                    nodes_with_multiple_parents,
                )
        if not nx.is_connected(graph.to_undirected()):
            print("The graph is disconnected, which violates the tree structure.")

    # Check if the graph is a DAG
    if nx.is_directed_acyclic_graph(graph):
        print("Graph is a directed acyclic graph (DAG)")
    else:
        print("Graph is not a directed acyclic graph")
        try:
            cycle = nx.find_cycle(graph)
            print("Cycle found:", cycle)
        except nx.NetworkXNoCycle:
            print("No explicit cycle detected, but the graph is still not acyclic.")

    # Convert to directed graph for further checks
    directed_graph = graph.to_directed()

    # Check if the directed graph is a tree
    if nx.is_tree(directed_graph):
        print("Graph is a directed tree")
    else:
        print("Graph is not a directed tree")
        nodes_with_multiple_parents = find_nodes_with_multiple_parents(directed_graph)
        if nodes_with_multiple_parents:
            print(
                "Nodes with multiple parents (violating directed tree structure):",
                nodes_with_multiple_parents,
            )

    # Check if the graph is an arborescence (rooted tree)
    if nx.is_arborescence(graph):
        print("Graph is an arborescence")
    else:
        print("Graph is not an arborescence")
        root_candidates = [node for node in graph.nodes() if graph.in_degree(node) == 0]
        if len(root_candidates) != 1:
            print(
                "The graph must have exactly one root (a node with no incoming edges). Candidates:",
                root_candidates,
            )
        if not nx.is_weakly_connected(graph):
            print(
                "The graph is not weakly connected, which violates the arborescence definition."
            )


def main():
    parser = argparse.ArgumentParser(description="Check if a graph is a tree or DAG")

    parser.add_argument(
        "input_file", type=str, help="Path to the input graph pickle file"
    )
    args = parser.parse_args()

    process_graph(args.input_file)


if __name__ == "__main__":
    main()
