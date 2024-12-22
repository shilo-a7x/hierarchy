import os
import subprocess
from natsort import natsorted

input_base_dir = os.path.join("data", "synthetic")
output_base_dir = os.path.join("output", "synthetic")

for num_nodes_dir in natsorted(os.listdir(output_base_dir)):
    output_dir = os.path.join(output_base_dir, num_nodes_dir)
    input_dir = os.path.join(input_base_dir, num_nodes_dir)

    if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
        continue

    for output_file in natsorted(os.listdir(output_dir)):
        if output_file.startswith("H_") and output_file.endswith(".pkl"):

            percentage = output_file.split("_")[-1].replace(".pkl", "")

            input_G_file = f"G_{num_nodes_dir}_{percentage}.pkl"
            input_G_path = os.path.join(input_dir, input_G_file)

            if os.path.exists(input_G_path):

                output_csv = "results.csv"

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
