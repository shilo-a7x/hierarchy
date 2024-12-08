import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_metric_vs_percentage(stat_name, df, output_dir):
    """
    Generates a figure for a single metric.
    Each figure has subplots (one per unique number of nodes).
    Within each subplot, plots the metric vs percentage for all scores.
    Includes one shared legend for all subplots in the figure.
    """
    # Extract unique number of nodes from the G_filename
    num_nodes_values = df["G_filename"].apply(lambda x: x.split("_")[1]).unique()
    num_nodes_values = sorted(num_nodes_values)  # Sort for consistency

    # Create subplots for each number of nodes
    num_subplots = len(num_nodes_values)
    rows = (num_subplots + 1) // 2  # Arrange approximately 2 columns per row
    cols = 2 if num_subplots > 1 else 1

    # Initialize the figure with the required number of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    axes = axes.flatten()  # Flatten the 2D array to 1D for easier indexing

    # Plot data into each subplot
    for idx, num_nodes in enumerate(num_nodes_values):
        # Filter DataFrame for the specific number of nodes
        filtered_df = df[df["G_filename"].apply(lambda x: x.split("_")[1] == num_nodes)]

        # Loop over scores and plot the comparisons
        for score in filtered_df["Score"].unique():
            # Further subset data for this score
            score_df = filtered_df[filtered_df["Score"] == score]

            # Extract percentage and corresponding statistic values
            percentages = score_df["H_filename"].apply(
                lambda x: int(x.split("_")[2].replace(".pkl", ""))
            )
            stat_values = score_df[stat_name]

            # Plotting on the current axis
            ax = axes[idx]
            ax.plot(percentages, stat_values, marker="o", label=score)

        # Customize subplot
        ax.set_title(f"Num Nodes = {num_nodes}")
        ax.set_xlabel("Percentage")
        ax.set_ylabel(stat_name)

    # Handle any remaining axes if the number of nodes is less than the number of subplots
    for idx in range(len(num_nodes_values), len(axes)):
        fig.delaxes(axes[idx])  # Remove unused axes

    # Add a single shared legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize="small",
        frameon=True,
    )

    # Adjust layout to prevent overlaps
    plt.tight_layout()

    # Save the figure for this metric
    metric_filename = os.path.join(output_dir, f"{stat_name}_vs_percentage.png")
    plt.savefig(metric_filename, bbox_inches="tight")  # Ensure all elements fit
    plt.close()


def generate_all_metric_plots(csv_file, output_dir):
    """
    Loops over all statistics in the CSV and generates individual plots.
    Each plot has subplots for different numbers of nodes.
    """
    # Load the CSV data
    df = pd.read_csv(csv_file)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine all statistic columns (excluding non-metric columns)
    stat_columns = [
        col for col in df.columns if col not in ["G_filename", "H_filename", "Score"]
    ]

    # Generate a plot for each statistic
    for stat in stat_columns:
        print(f"Generating plots for: {stat}")
        plot_metric_vs_percentage(stat, df, output_dir)


if __name__ == "__main__":
    # Path to the CSV file
    csv_file = "results.csv"

    # Directory to save the plots
    output_dir = "plots"

    # Generate all plots
    generate_all_metric_plots(csv_file, output_dir)
