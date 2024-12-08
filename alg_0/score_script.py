import os
import subprocess
from natsort import natsorted
import json


def load_scores_from_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as f:
        config = json.load(f)
    if "score_functions" not in config:
        raise KeyError("The configuration file must contain a 'score_functions' key.")
    return config["score_functions"].keys()


input_base_dir = os.path.join("data", "synthetic")
output_base_dir = os.path.join("output", "synthetic")

config_path = "config.json"

try:
    all_scores = load_scores_from_config(config_path)
except (FileNotFoundError, KeyError) as e:
    print(f"Error: {e}")
    exit(1)

for num_nodes_dir in natsorted(os.listdir(input_base_dir)):
    input_dir = os.path.join(input_base_dir, num_nodes_dir)
    output_dir = os.path.join(output_base_dir, num_nodes_dir)
    os.makedirs(output_dir, exist_ok=True)

    for input_file in natsorted(os.listdir(input_dir)):

        if input_file.startswith("G_") and input_file.endswith(".pkl"):

            percent = input_file.split("_")[-1].replace(".pkl", "")

            input_path = os.path.join(input_dir, input_file)

            for score in all_scores:

                output_file = f"H_{num_nodes_dir}_{percent}_{score}.pkl"
                output_path = os.path.join(output_dir, output_file)

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
