import pickle
import argparse
import networkx as nx
import leidenalg as la
import igraph as ig


def convert_to_igraph(graph: nx.Graph):
    ig_graph = ig.Graph(directed=graph.is_directed())
    ig_graph.add_vertices(list(graph.nodes()))
    ig_graph.add_edges(list(graph.edges()))
    return ig_graph


def perform_leiden(graph: nx.Graph):
    ig_graph = convert_to_igraph(graph)
    partition = la.find_partition(ig_graph, partition_type=la.ModularityVertexPartition)
    communities = {}
    for comm, nodes in enumerate(partition):
        for node in nodes:
            communities[node] = comm
    return communities


def alg(graph: nx.Graph):
    communities = perform_leiden(graph)
    edges_to_remove = []
    for u, v in graph.edges:
        if communities[u] != communities[v]:
            edges_to_remove.append((u, v))

    graph.remove_edges_from(edges_to_remove)

    return graph


def load_graph(file_path):
    with open(file_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def save_graph(graph, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved tree to {file_path}")


def process_graph(input_file, output_file):
    graph = load_graph(input_file)
    result = alg(graph)
    save_graph(result, output_file)


def main():
    parser = argparse.ArgumentParser(description="Process a graph")

    parser.add_argument(
        "input_file", type=str, help="Path to the input graph pickle file"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the output graph pickle file",
    )
    args = parser.parse_args()
    process_graph(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
