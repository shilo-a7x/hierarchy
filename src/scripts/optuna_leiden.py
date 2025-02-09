import pickle
import networkx as nx
import leidenalg as la
import igraph as ig
import optuna
import csv
import json
import argparse


def load_graph(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def convert_to_igraph(graph: nx.Graph):
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    ig_graph = ig.Graph(directed=graph.is_directed())
    ig_graph.add_vertices(len(node_mapping))
    ig_graph.add_edges([(node_mapping[u], node_mapping[v]) for u, v in graph.edges()])
    return ig_graph, reverse_mapping


def perform_leiden(graph: nx.Graph, resolution: float):
    ig_graph, reverse_mapping = convert_to_igraph(graph)
    partition = la.find_partition(
        ig_graph, la.RBConfigurationVertexPartition, resolution_parameter=resolution
    )
    return {
        reverse_mapping[node]: comm
        for comm, nodes in enumerate(partition)
        for node in nodes
    }


def optimize_leiden(trial, G, T):
    resolution = trial.suggest_float("resolution", 0.1, 2.0)
    communities = perform_leiden(G, resolution)
    edges_to_remove = [(u, v) for u, v in G.edges if communities[u] != communities[v]]
    S = G.copy()
    S.remove_edges_from(edges_to_remove)
    return compute_f1_score(G, T, S)


def compute_f1_score(G: nx.Graph, T: nx.Graph, S: nx.Graph):
    TP = sum(1 for edge in T.edges if S.has_edge(*edge))
    FP = sum(1 for edge in G.edges if edge not in T.edges and S.has_edge(*edge))
    FN = sum(1 for edge in T.edges if not S.has_edge(*edge))
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    return (
        (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Leiden algorithm using Optuna"
    )
    parser.add_argument("graph_name", type=str, help="Name of the graph")
    parser.add_argument(
        "G_filepath", type=str, help="Path to the input graph pickle file"
    )
    parser.add_argument(
        "T_filepath", type=str, help="Path to the ground truth graph pickle file"
    )
    parser.add_argument(
        "output_json",
        type=str,
        help="Path to save the best Leiden parameters JSON file",
    )
    parser.add_argument(
        "output_csv", type=str, help="Path to save the graph results CSV file"
    )
    args = parser.parse_args()

    G = load_graph(args.G_filepath)
    T = load_graph(args.T_filepath)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_leiden(trial, G, T), n_trials=50)

    best_params = study.best_params
    best_f1 = study.best_value

    with open(args.output_json, "w") as f:
        json.dump(
            {"Best Parameters": best_params, "Best F1-Score": best_f1}, f, indent=4
        )

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Graph Name", "Nodes", "Edges", "Best F1-Score"])
        writer.writerow(
            [
                args.graph_name,
                G.number_of_nodes(),
                G.number_of_edges(),
                round(best_f1, 4),
            ]
        )

    print("Optimization complete. Results saved.")


if __name__ == "__main__":
    main()
