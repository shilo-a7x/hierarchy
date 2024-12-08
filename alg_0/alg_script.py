import os
import subprocess
from natsort import natsorted

input_base_dir = os.path.join("data", "synthetic")
output_base_dir = os.path.join("output", "synthetic")

for num_nodes_dir in natsorted(os.listdir(input_base_dir)):
    input_dir = os.path.join(input_base_dir, num_nodes_dir)
    output_dir = os.path.join(output_base_dir, num_nodes_dir)
    os.makedirs(output_dir, exist_ok=True)

    for input_file in natsorted(os.listdir(input_dir)):

        if input_file.startswith("G_") and input_file.endswith(".pkl"):

            percent = input_file.split("_")[-1].replace(".pkl", "")

            input_path = os.path.join(input_dir, input_file)
            output_file = f"H_{num_nodes_dir}_{percent}.pkl"
            output_path = os.path.join(output_dir, output_file)

            print(f"Processing {input_file} -> {output_file}")
            subprocess.run(["python", "alg.py", input_path, output_path])
