import os
import subprocess
from natsort import natsorted
import json


# Load the scores from the config.json file
def load_scores_from_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as f:
        config = json.load(f)
    if "score_functions" not in config:
        raise KeyError("The configuration file must contain a 'score_functions' key.")
    return config["score_functions"].keys()


# Define input and output base directories
input_base_dir = os.path.join("data", "synthetic")
output_base_dir = os.path.join("output", "synthetic")

# Path to the config file
config_path = "config.json"

# Load all possible score functions
try:
    all_scores = load_scores_from_config(config_path)
except (FileNotFoundError, KeyError) as e:
    print(f"Error: {e}")
    exit(1)

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

            # Construct input file path
            input_path = os.path.join(input_dir, input_file)

            # Run the algorithm for each score
            for score in all_scores:
                # Construct the output file name and path
                output_file = f"H_{num_nodes_dir}_{percent}_{score}.pkl"
                output_path = os.path.join(output_dir, output_file)

                # Run the algorithm
                print(f"Processing {input_file} with score '{score}' -> {output_file}")
                subprocess.run(
                    [
                        "python",
                        "alg.py",
                        input_path,
                        output_path,
                        "--score",
                        score,
                        "--config",
                        config_path,
                    ]
                )
