import pickle
import networkx as nx
import argparse


def dag_from_score(graph, score) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(graph.nodes)
    for u, v in graph.edges:
        if score[u] > score[v]:
            dag.add_edge(u, v)
        # elif score[u] < score[v]:
        #     dag.add_edge(v, u)
    return dag


def alg(graph: nx.Graph):
    graph_score = nx.betweenness_centrality(graph.to_undirected())
    dag = dag_from_score(graph, graph_score)
    return dag


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
    result_tree = alg(graph)
    save_graph(result_tree, output_file)


def main():
    parser = argparse.ArgumentParser(description="Process a graph and output a tree")

    parser.add_argument(
        "input_file", type=str, help="Path to the input graph pickle file"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the output tree pickle file",
    )

    args = parser.parse_args()

    process_graph(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
