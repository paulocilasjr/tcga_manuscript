import pandas as pd

def count_zeros_and_ones(csv_file):
    """
    Counts the number of 0s and 1s in the 'label' column of a CSV file.

    Args:
        csv_file (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the counts of 0s and 1s, or None if an error occurs.
    """
    try:
        df = pd.read_csv(csv_file)
        count_0 = (df['label'] == 0).sum()
        count_1 = (df['label'] == 1).sum()
        return count_0, count_1
    except FileNotFoundError:
        print(f"Error: File not found - {csv_file}")
        return None
    except KeyError:
        print("Error: 'label' column not found in the CSV file.")
        return None
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        return None

if __name__ == "__main__":
    file_path = "tcga_embedding_resnet50_label.csv"  # Replace with your CSV file path
    counts = count_zeros_and_ones(file_path)

    if counts:
        count_0, count_1 = counts
        print(f"Number of 0s: {count_0}")
        print(f"Number of 1s: {count_1}")
