import os
import pandas as pd
import sys

def process_directories_and_match(directory_path, metadata_csv_path):
    """
    Lists directories, matches them with sample names from a CSV,
    and counts matches and non-matches.
    """
    try:
        directories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return None, None, None

    try:
        metadata_df = pd.read_csv(metadata_csv_path)
    except FileNotFoundError:
        print(f"Error: Metadata CSV file not found at {metadata_csv_path}")
        return None, None, None

    sample_names = metadata_df['sample_name'].tolist()

    matched_count = 0
    unmatched_count = 0
    matched_samples = set()
    unmatched_samples = []

    for sample_name in sample_names:
        found_match = any(sample_name in dir_name for dir_name in directories)
        if found_match:
            matched_count += 1
            matched_samples.add(sample_name)
        else:
            unmatched_count += 1
            unmatched_samples.append(sample_name)

    return matched_count, unmatched_count, unmatched_samples

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <directory_path> <metadata_csv_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    metadata_csv_path = sys.argv[2]
    matched, unmatched, unmatched_sample_list = process_directories_and_match(directory_path, metadata_csv_path)

    if matched is not None and unmatched is not None:
        print(f"Matched sample names: {matched}")
        print(f"Unmatched sample names: {unmatched}")
        if unmatched > 0:
            print("Unmatched Sample List:")
            for sample in unmatched_sample_list:
                pass
                #print(sample)

