import os
import subprocess
from natsort import natsorted

# Define input and output base directories
input_base_dir = os.path.join("data", "synthetic")
output_base_dir = os.path.join("output", "synthetic")

# Iterate through all subdirectories in the input directory
for num_nodes_dir in natsorted(os.listdir(input_base_dir)):
    input_dir = os.path.join(input_base_dir, num_nodes_dir)
    output_dir = os.path.join(output_base_dir, num_nodes_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all input graph files
    for input_file in natsorted(os.listdir(input_dir)):
        # Process only files matching the pattern G_{num_of_nodes}_{percentage}.pkl
        if input_file.startswith("G_") and input_file.endswith(".pkl"):
            # Extract the `{percentage}` part of the file name
            percent = input_file.split("_")[-1].replace(".pkl", "")

            # Construct input and output file paths
            input_path = os.path.join(input_dir, input_file)
            output_file = f"H_{num_nodes_dir}_{percent}.pkl"
            output_path = os.path.join(output_dir, output_file)

            # Run the algorithm
            print(f"Processing {input_file} -> {output_file}")
            subprocess.run(["python", "alg.py", input_path, output_path])
