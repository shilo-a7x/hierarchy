import os
import subprocess
from natsort import natsorted

# Define the base input and output directories
input_base_dir = os.path.join("data", "synthetic")
output_base_dir = os.path.join("output", "synthetic")

# Iterate through the subdirectories in the output directory (for each graph size: 100, 1000)
for num_nodes_dir in natsorted(os.listdir(output_base_dir)):
    output_dir = os.path.join(output_base_dir, num_nodes_dir)
    input_dir = os.path.join(input_base_dir, num_nodes_dir)

    # Check if both input and output directories exist for the current size
    if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
        continue

    # Iterate through all the H_{num}_{percentage}.pkl files in the output directory
    for output_file in natsorted(os.listdir(output_dir)):
        if output_file.startswith("H_") and output_file.endswith(".pkl"):
            # Extract the percentage from the H file name (e.g., "H_100_11.pkl")
            parts = output_file.split("_")
            num_nodes = parts[1]  # Number of nodes, e.g., "100"
            # percent = parts[2].replace(".pkl", "")  # Percentage, e.g., "11"

            # Construct the corresponding G file name (we now always want G_{num}_0.pkl)
            input_G_file = f"G_{num_nodes}_0.pkl"  # Always use percentage 0 for G file
            input_G_path = os.path.join(input_dir, input_G_file)

            # Check if the corresponding G file exists
            if os.path.exists(input_G_path):
                # Construct the output CSV file path
                output_csv = "results.csv"

                # Run the compare_graphs.py script for the current pair
                input_H_path = os.path.join(output_dir, output_file)
                print(f"Comparing {input_G_path} and {input_H_path}")
                subprocess.run(
                    [
                        "python",
                        "compare_graphs.py",
                        input_G_path,
                        input_H_path,
                        output_csv,
                    ]
                )
            else:
                print(
                    f"Warning: Corresponding G file for {output_file} not found at {input_G_path}."
                )
