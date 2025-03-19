import pandas as pd
import sys

def count_bag_labels(csv_file):
    """
    Loads a CSV file, extracts the 'bag_label' column, and counts the occurrences of 0 and 1.

    Args:
        csv_file (str): Path to the CSV file.
    """
    try:
        df = pd.read_csv(csv_file)

        if 'bag_label' not in df.columns:
            print("Error: 'bag_label' column not found in the CSV file.")
            return

        bag_labels = df['bag_label']

        zeros_count = (bag_labels == 0).sum()
        ones_count = (bag_labels == 1).sum()

        print(f"Zeros: {zeros_count}")
        print(f"Ones: {ones_count}")

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <input_file.csv>")
        sys.exit(1)

    input_file = sys.argv[1]

    count_bag_labels(input_file)
