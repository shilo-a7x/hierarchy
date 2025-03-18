import networkx as nx
import argparse
import pickle
import matplotlib.pyplot as plt


def load_graph(file_path):
    with open(file_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def compute_tp_fn_positive_vs_depth(T: nx.Graph, S: nx.Graph):
    root = [node for node, degree in T.in_degree() if degree == 0]  # Find root nodes
    if root:
        root = root[0]
    else:
        raise ValueError("Tree T does not have a clear root node.")

    depth_dict = nx.single_source_shortest_path_length(T, root)

    tp_vs_depth = {}
    fn_vs_depth = {}
    positive_vs_depth = {}
    for node, depth in depth_dict.items():
        tp = sum(1 for neighbor in T.neighbors(node) if S.has_edge(node, neighbor))
        fn = sum(1 for neighbor in T.neighbors(node) if not S.has_edge(node, neighbor))
        positive = tp + fn

        tp_vs_depth[depth] = tp_vs_depth.get(depth, 0) + tp
        fn_vs_depth[depth] = fn_vs_depth.get(depth, 0) + fn
        positive_vs_depth[depth] = positive_vs_depth.get(depth, 0) + positive

    return tp_vs_depth, fn_vs_depth, positive_vs_depth


def plot_tp_fn_positive_vs_depth(graph_name, T_filepath, S_filepath, output_plot):
    T = load_graph(T_filepath)
    S = load_graph(S_filepath)

    tp_vs_depth, fn_vs_depth, positive_vs_depth = compute_tp_fn_positive_vs_depth(T, S)

    depths = sorted(tp_vs_depth.keys())
    tp_values = [tp_vs_depth[d] for d in depths]
    fn_values = [fn_vs_depth[d] for d in depths]
    positive_values = [positive_vs_depth[d] for d in depths]

    plt.figure(figsize=(8, 6))
    plt.plot(depths, tp_values, marker="o", linestyle="-", label="True Positives (TP)")
    plt.plot(
        depths, fn_values, marker="s", linestyle="--", label="False Negatives (FN)"
    )
    plt.plot(
        depths, positive_values, marker="^", linestyle=":", label="Total Positives"
    )
    plt.xlabel("Depth")
    plt.ylabel("Count")
    plt.title(f"TP, FN & Positives vs Depth Plot for {graph_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_plot)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot TP vs Depth from graphs.")
    parser.add_argument("graph_name", type=str, help="Name of the graph.")
    parser.add_argument(
        "T_filepath", type=str, help="Path to the ground truth tree file (T)."
    )
    parser.add_argument(
        "S_filepath", type=str, help="Path to the approximation tree file (S)."
    )
    parser.add_argument("output_plot", type=str, help="Path to save the output plot.")

    args = parser.parse_args()

    plot_tp_fn_positive_vs_depth(
        args.graph_name, args.T_filepath, args.S_filepath, args.output_plot
    )


if __name__ == "__main__":
    main()
