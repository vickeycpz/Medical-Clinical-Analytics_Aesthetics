import os
import pandas as pd
import kagglehub

def main():
    print("Downloading dataset from Kaggle...")

    # Download dataset
    path = kagglehub.dataset_download("joniarroba/noshowappointments")

    print(f"\nDataset downloaded to:\n{path}")

    # List files
    files = os.listdir(path)

    print("\nFiles found:")
    for file in files:
        print(f" - {file}")

    # Try loading CSV automatically
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        print("\nNo CSV file found.")
        return

    csv_path = os.path.join(path, csv_files[0])

    print(f"\nLoading CSV file:\n{csv_path}")

    df = pd.read_csv(csv_path)

    print("\nDataset Preview:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nBasic Statistics:")
    print(df.describe(include="all"))

    # Save a local working copy
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "appointments_raw.csv")

    df.to_csv(output_path, index=False)

    print(f"\nLocal copy saved to:\n{output_path}")

if __name__ == "__main__":
    main()