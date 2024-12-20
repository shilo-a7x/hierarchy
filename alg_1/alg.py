import os
import pickle
import networkx as nx
import json


def alg(graph: nx.Graph):
    # graph_score = graph_nodes_score(graph, score_type, score_functions)
    # dag = dag_from_score(graph, graph_score)
    # communities = louvain_communities(dag.to_undirected(), threshold=0.8)
    # community_subgraphs = [dag.subgraph(community).copy() for community in communities]
    # hierarchy_tree = tree_from_subdags(community_subgraphs, graph_score)
    # print(nx.is_tree(hierarchy_tree))
    # return hierarchy_tree
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
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="config.json",
    #     help="Path to the configuration JSON file (default: 'config.json')",
    # )

    args = parser.parse_args()

    # config = load_config(args.config)

    process_graph(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
