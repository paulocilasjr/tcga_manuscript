import pandas as pd
import sys

def find_unique_samples(file1_path, file2_path):
    """
    Finds sample names in file_1 that are not present in file_2.

    Args:
        file1_path (str): Path to file_1.csv.
        file2_path (str): Path to file_2.csv.

    Returns:
        list: A list of sample names unique to file_1.
    """
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        samples1 = set(df1['sample_name'])
        samples2 = set(df2['sample_name'])

        unique_samples = list(samples1 - samples2)
        return unique_samples

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return []
    except KeyError:
        print("Error: 'sample_name' column not found in one or both CSV files.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py file_1.csv file_2.csv")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]

    unique_samples = find_unique_samples(file1_path, file2_path)

    if unique_samples:
        print("Sample names in file_1.csv but not in file_2.csv:")
        for sample in unique_samples:
            print(sample)
    else:
        print("No unique samples found in file_1.csv or an error occurred.")
