import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_metric_vs_percentage(stat_name, df, output_dir):
    num_nodes_values = df["Number of nodes"].unique()
    num_nodes_values = sorted(num_nodes_values)

    num_subplots = len(num_nodes_values)
    rows = (num_subplots + 1) // 2
    cols = 2 if num_subplots > 1 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    axes = axes.flatten()

    for idx, num_nodes in enumerate(num_nodes_values):
        filtered_df = df[df["Number of nodes"] == num_nodes]

        percentages = filtered_df["Added edges perncetage"].astype(int)
        stat_values = filtered_df[stat_name]

        ax = axes[idx]
        ax.plot(
            percentages,
            stat_values,
            marker="o",
            label=f"Num Nodes = {num_nodes}",
        )

        ax.set_title(f"Num Nodes = {num_nodes}")
        ax.set_xlabel("Percentage")
        ax.set_ylabel(stat_name)

    for idx in range(len(num_nodes_values), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    metric_filename = os.path.join(output_dir, f"{stat_name} VS Percentage.png")
    plt.savefig(metric_filename, bbox_inches="tight")
    plt.close()


def generate_all_metric_plots(csv_file, output_dir):
    df = pd.read_csv(csv_file)

    os.makedirs(output_dir, exist_ok=True)

    stat_columns = [
        col
        for col in df.columns
        if col
        not in ["G filename", "H filename", "Number of nodes", "Added edges perncetage"]
    ]

    for stat in stat_columns:
        print(f"Generating plots for: {stat}")
        plot_metric_vs_percentage(stat, df, output_dir)


if __name__ == "__main__":

    csv_file = "results.csv"
    output_dir = "plots"

    generate_all_metric_plots(csv_file, output_dir)
