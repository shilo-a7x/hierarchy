import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import numpy as np


def plot_metric_vs_percentage(stat_name, df, output_dir):
    num_nodes_values = df["G_filename"].apply(lambda x: x.split("_")[1]).unique()
    num_nodes_values = sorted(num_nodes_values)

    num_subplots = len(num_nodes_values)
    rows = (num_subplots + 1) // 2
    cols = 2 if num_subplots > 1 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    axes = axes.flatten()

    unique_scores = df["Score"].unique()
    num_colors = len(unique_scores)

    colors = cm.tab10(np.linspace(0, 1, num_colors))

    markers = ["o", "s", "D", "^", "v"]

    score_to_style = {
        score: {
            "color": colors[idx % len(colors)],
            "marker": markers[idx % len(markers)],
        }
        for idx, score in enumerate(unique_scores)
    }

    for idx, num_nodes in enumerate(num_nodes_values):

        filtered_df = df[df["G_filename"].apply(lambda x: x.split("_")[1] == num_nodes)]

        for score in filtered_df["Score"].unique():

            score_df = filtered_df[filtered_df["Score"] == score]

            percentages = score_df["H_filename"].apply(
                lambda x: int(x.split("_")[2].replace(".pkl", ""))
            )
            stat_values = score_df[stat_name]

            ax = axes[idx]
            ax.plot(
                percentages,
                stat_values,
                marker=score_to_style[score]["marker"],
                label=score,
                color=score_to_style[score]["color"],
            )

        ax.set_title(f"Num Nodes = {num_nodes}")
        ax.set_xlabel("Percentage")
        ax.set_ylabel(stat_name)

    for idx in range(len(num_nodes_values), len(axes)):
        fig.delaxes(axes[idx])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize="small",
        frameon=True,
    )

    plt.tight_layout()

    metric_filename = os.path.join(output_dir, f"{stat_name}_vs_percentage.png")
    plt.savefig(metric_filename, bbox_inches="tight")
    plt.close()


def generate_all_metric_plots(csv_file, output_dir):
    df = pd.read_csv(csv_file)

    os.makedirs(output_dir, exist_ok=True)

    stat_columns = [
        col for col in df.columns if col not in ["G_filename", "H_filename", "Score"]
    ]

    for stat in stat_columns:
        print(f"Generating plots for: {stat}")
        plot_metric_vs_percentage(stat, df, output_dir)


if __name__ == "__main__":

    csv_file = "results.csv"

    output_dir = "plots"

    generate_all_metric_plots(csv_file, output_dir)
