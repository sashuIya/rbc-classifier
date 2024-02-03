import os

import pandas as pd

# Specify the folder path containing the CSV files
folder_path = "dataset/1000"

# Define the mapping for renaming 'y' column values
value_mapping = {
    # "unlabeled": "unlabeled",
    # "wrong": "wrong",
    # "0": "red blood cell",
    # "1": "spheroid cell",
    # "2": "echinocyte",
    # Add more mappings as needed
}


def check_values(df):
    result = True
    for k in df["y"].unique():
        if k not in value_mapping.keys():
            print("{} not in value_mapping".format(k))
            result = False

    return result


# Iterate through each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith("_features.csv"):
        file_path = os.path.join(folder_path, filename)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Rename 'y' column values using the specified mapping
        if not check_values(df):
            break

        df["y"] = df["y"].map(value_mapping)

        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path, index=False)

print("Conversion complete.")
