import networkx as nx
import argparse
import csv
import pickle
import os


def load_graph(file_path):
    with open(file_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def compute_statistics(G: nx.Graph, H: nx.Graph):
    TP = FP = FN = TN = 0

    for u, v, data in G.edges(data=True):
        if data["type"] == "hierarchy":
            if H.has_edge(u, v):
                TP += 1
            else:
                FN += 1
        elif data["type"] == "non-hierarchy":
            if H.has_edge(u, v):
                FP += 1
            else:
                TN += 1

    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return TP, FP, FN, TN, accuracy, precision, recall, f1_score


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
        TN,
        accuracy,
        precision,
        recall,
        f1_score,
    ) = compute_statistics(G, H)

    H_filename = os.path.basename(H_filepath).replace(".pkl", "")
    G_filename = H_filename.replace("H", "G")

    parts = H_filename.split("_")
    num_of_nodes = parts[1]
    percentage = parts[2]

    total_edges = len(G.edges())

    result_data = {
        "G filename": G_filename,
        "H filename": H_filename,
        "Number of nodes": num_of_nodes,
        "Added edges perncetage": percentage,
        "TP": round(TP / total_edges, 4),
        "FP": round(FP / total_edges, 4),
        "FN": round(FN / total_edges, 4),
        "TN": round(TN / total_edges, 4),
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1_score, 4),
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
