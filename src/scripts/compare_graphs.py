import networkx as nx
import argparse
import csv
import pickle


def load_graph(file_path):
    with open(file_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def compute_statistics(G: nx.Graph, T: nx.Graph, S: nx.Graph):
    TP = FP = FN = TN = 0
    positive_edges = set(T.edges)
    total_edges = set(G.edges)
    negative_edges = total_edges - positive_edges
    illegal_edges = set(S.edges()) - total_edges

    for edge in total_edges:
        u, v = edge
        if edge in positive_edges:
            if S.has_edge(u, v):
                TP += 1
            else:
                FN += 1
        elif edge in negative_edges:
            if S.has_edge(u, v):
                FP += 1
            else:
                TN += 1

    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return (
        TP,
        FP,
        FN,
        TN,
        accuracy,
        precision,
        NPV,
        recall,
        TNR,
        f1_score,
        len(illegal_edges),
    )


def compare_graphs(graph_name, G_filepath, T_filepath, S_filepath, output_csv):
    G = load_graph(G_filepath)
    T = load_graph(T_filepath)
    S = load_graph(S_filepath)

    if set(G.nodes()) != set(S.nodes()):
        print("Error: Graphs have different nodes.")
        return

    (
        TP,
        FP,
        FN,
        TN,
        accuracy,
        precision,
        NPV,
        recall,
        TNR,
        f1_score,
        shortcut_edges,
    ) = compute_statistics(G, T, S)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    result_data = {
        "Graph name": graph_name,
        "Nodes": num_nodes,
        "Edges": num_edges,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Shortcut edges": shortcut_edges,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Negative predictive value": round(NPV, 4),
        "Recall": round(recall, 4),
        "True negative rate": round(TNR, 4),
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
    parser.add_argument("graph_name", type=str, help="name of the graph.")
    parser.add_argument(
        "G_filepath", type=str, help="Path to the ground truth graph file (G)."
    )
    parser.add_argument(
        "T_filepath", type=str, help="Path to the ground truth tree file (T)."
    )
    parser.add_argument(
        "S_filepath", type=str, help="Path to the approximation tree file (S)."
    )
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file.")

    args = parser.parse_args()

    compare_graphs(
        args.graph_name,
        args.G_filepath,
        args.T_filepath,
        args.S_filepath,
        args.output_csv,
    )


if __name__ == "__main__":
    main()
