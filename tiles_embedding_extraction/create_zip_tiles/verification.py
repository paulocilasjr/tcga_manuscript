import zipfile
import pandas as pd
import sys
import os

def process_zip_and_match(zip_path, metadata_csv_path):
    """
    Loads a zip file, extracts filenames, matches them with sample names from a CSV,
    and counts matches and non-matches.
    """

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            filenames = [f.filename for f in zip_ref.infolist()]

    except FileNotFoundError:
        print(f"Error: Zip file not found at {zip_path}")
        return None, None

    try:
        metadata_df = pd.read_csv(metadata_csv_path)
    except FileNotFoundError:
        print(f"Error: Metadata CSV file not found at {metadata_csv_path}")
        return None, None

    sample_names = metadata_df['sample_name'].tolist()
    print(len(sample_names))

    matched_count = 0
    unmatched_count = 0
    matched_samples = set()
    unmatched_samples = []

    for sample_name in sample_names:
        found_match = False
        for filename in filenames:
            if sample_name in filename and sample_name not in matched_samples:
                matched_count += 1
                matched_samples.add(sample_name)
                found_match = True
                break

        if not found_match and sample_name not in matched_samples:
            unmatched_count += 1
            unmatched_samples.append(sample_name)

    return matched_count, unmatched_count, unmatched_samples

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <zip_file_path> <metadata_csv_path>")
        sys.exit(1)

    zip_file_path = sys.argv[1]
    metadata_csv_path = sys.argv[2]

    matched, unmatched, unmatched_sample_list = process_zip_and_match(zip_file_path, metadata_csv_path)

    if matched is not None and unmatched is not None:
        print(f"Matched sample names: {matched}")
        print(f"Unmatched sample names: {unmatched}")
        if unmatched > 0:
          print("Unmatched Sample List:")
          for sample in unmatched_sample_list:
            print(sample)
