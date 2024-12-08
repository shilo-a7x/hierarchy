import networkx as nx
import argparse
import csv
import pickle
import matplotlib.pyplot as plt
import os


def load_graph(file_path):
    with open(file_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def compute_statistics(G, H):
    G_edges = set(G.edges())
    H_edges = set(H.edges())

    TP = len(G_edges & H_edges)
    FP = len(H_edges - G_edges)
    FN = len(G_edges - H_edges)
    reversed_edges = set((v, u) for u, v in G_edges) & H_edges
    REV = len(reversed_edges)

    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    G_roots = [v for v in G.nodes if G.in_degree(v) == 0]
    H_roots = [v for v in H.nodes if H.in_degree(v) == 0]

    if len(G_roots) == 1 and len(H_roots) == 1:
        G_root = G_roots[0]
        H_root = H_roots[0]

        G_depths = nx.shortest_path_length(G, source=G_root)
        H_depths = nx.shortest_path_length(H, source=H_root)

        depth_deviation = sum(
            abs(G_depths.get(node, 0) - H_depths.get(node, 0)) for node in G.nodes()
        )

        mean_depth_error = depth_deviation / len(G.nodes()) if len(G.nodes()) > 0 else 0
    else:

        depth_deviation = 0
        mean_depth_error = 0

    return (
        TP,
        FP,
        FN,
        REV,
        accuracy,
        precision,
        recall,
        f1_score,
        depth_deviation,
        mean_depth_error,
    )


def compare_graphs(G_filepath, H_filepath, output_csv):
    G = load_graph(G_filepath)
    H = load_graph(H_filepath)

    if set(G.nodes()) != set(H.nodes()):
        print("Error: Graphs have different nodes.")
        return

    (
        TP,
        FP,
        FN,
        REV,
        accuracy,
        precision,
        recall,
        f1_score,
        depth_deviation,
        mean_depth_error,
    ) = compute_statistics(G, H)

    H_filename = os.path.basename(H_filepath).replace(".pkl", "")
    G_filename = H_filename.replace("H", "G")

    parts = H_filename.split("_")
    score = "_".join(parts[3:]) if len(parts) > 3 else "default"

    result_data = {
        "G_filename": G_filename,
        "H_filename": H_filename,
        "Score": score,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Reversed Edges": REV,
        "Accuracy": round(accuracy, 6),
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1-Score": round(f1_score, 6),
        "Depth Deviation": round(depth_deviation, 6),
        "Mean Depth Error": round(mean_depth_error, 6),
    }

    with open(output_csv, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_data.keys())

        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(result_data)


def main():

    parser = argparse.ArgumentParser(
        description="Compare two graphs and compute statistics."
    )
    parser.add_argument(
        "G_filepath", type=str, help="Path to the ground truth graph file (G)."
    )
    parser.add_argument(
        "H_filepath", type=str, help="Path to the approximation graph file (H)."
    )
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file.")

    args = parser.parse_args()

    compare_graphs(args.G_filepath, args.H_filepath, args.output_csv)


if __name__ == "__main__":
    main()
