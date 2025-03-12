import os
import h5py
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm  # For progress tracking

def extract_embeddings(h5_file):
    """Extract embeddings from an HDF5 file."""
    with h5py.File(h5_file, 'r') as f:
        features = f['/features'][:]  # Load all features at once
    return features

def get_sample_name_from_filename(filename):
    """Extract core sample name from HDF5 filename."""
    parts = filename.split('-')[:4]
    return '-'.join(parts)

def process_batches(input_dir, metadata_file, output_pytorch, output_ludwig, batch_size=1000):
    """Iterate over batch directories and extract embeddings efficiently."""
    # Load metadata once
    metadata_df = pd.read_csv(metadata_file)
    metadata_dict = metadata_df.set_index('sample_name')['label'].to_dict()

    # Define column names
    pytorch_columns = ['sample_name', 'label'] + [f'feature_{i+1}' for i in range(1536)]
    ludwig_columns = ['sample_name', 'label', 'embedding']

    # Accumulate data in lists for efficiency
    pytorch_data = []
    ludwig_data = []

    # Process batch folders
    batch_folders = [f for f in sorted(os.listdir(input_dir)) if f.startswith('batch_')]
    for batch_folder in tqdm(batch_folders, desc="Processing batches"):
        features_path = os.path.join(input_dir, batch_folder, '20x_256px_0px_overlap/features_uni_v2')
        if not os.path.exists(features_path):
            continue

        # Process each .h5 file in the batch
        h5_files = [f for f in sorted(os.listdir(features_path)) if f.endswith('.h5')]

        for h5_file in tqdm(h5_files, desc=f"Processing {batch_folder}", leave=False):
            h5_path = os.path.join(features_path, h5_file)
            sample_name = get_sample_name_from_filename(h5_file)

            # Find metadata key that is a substring of sample_name
            matching_keys = [key for key in metadata_dict if key in sample_name]

            # Use the first matching key (if found), otherwise 'Unknown'
            label = metadata_dict[matching_keys[0]] if matching_keys else 'Unknown'

            embeddings = extract_embeddings(h5_path)

            # Collect all embeddings for this .h5 file
            for vector in embeddings:
                pytorch_row = [sample_name, label] + vector.tolist()
                ludwig_row = [sample_name, label, vector]  # Keep as array for now
                pytorch_data.append(pytorch_row)
                ludwig_data.append(ludwig_row)

                # Write in batches to avoid memory overload
                if len(pytorch_data) >= batch_size:
                    write_batch(pytorch_data, pytorch_columns, output_pytorch)
                    write_ludwig_batch(ludwig_data, ludwig_columns, output_ludwig)
                    pytorch_data.clear()
                    ludwig_data.clear()

    # Write any remaining data
    if pytorch_data:
        write_batch(pytorch_data, pytorch_columns, output_pytorch)
        write_ludwig_batch(ludwig_data, ludwig_columns, output_ludwig)

def write_batch(data, columns, output_file):
    """Write a batch of data to CSV."""
    df = pd.DataFrame(data, columns=columns)
    mode = 'a' if os.path.exists(output_file) else 'w'
    header = not os.path.exists(output_file)
    df.to_csv(output_file, mode=mode, header=header, index=False)

def write_ludwig_batch(data, columns, output_file):
    """Write a batch of Ludwig data to CSV with string conversion."""
    # Convert embedding vectors to space-separated strings
    processed_data = [[row[0], row[1], ' '.join(map(str, row[2]))] for row in data]
    df = pd.DataFrame(processed_data, columns=columns)
    mode = 'a' if os.path.exists(output_file) else 'w'
    header = not os.path.exists(output_file)
    df.to_csv(output_file, mode=mode, header=header, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract embeddings from HDF5 files.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing batch folders')
    parser.add_argument('--metadata_file', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--output_pytorch', type=str, required=True, help='Path to save PyTorch-style CSV')
    parser.add_argument('--output_ludwig', type=str, required=True, help='Path to save Ludwig-style CSV')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of rows per batch write')
    args = parser.parse_args()

    process_batches(args.input_dir, args.metadata_file, args.output_pytorch, args.output_ludwig, args.batch_size)
