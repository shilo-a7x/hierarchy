import pandas as pd

# Load the CSV file
input_file = "results_backup.csv"  # Replace with your actual file path
output_file = "results.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file)


# Fix the 'Score' column using the H_filename column
def extract_score(row):
    if len(row["H_filename"].split("_")) > 3:
        # Score extraction logic for meaningful filenames
        return "_".join(row["H_filename"].split("_")[3:])
    else:
        # Leave "default" untouched
        return row["Score"]


df["Score"] = df.apply(extract_score, axis=1)

# Save the fixed DataFrame to a new CSV
df.to_csv(output_file, index=False)

print(f"Fixed file saved to {output_file}")
