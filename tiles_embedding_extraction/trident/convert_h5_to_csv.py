import os
import h5py
import pandas as pd
import numpy as np
import argparse

def extract_embeddings(h5_file):
    """Extract embeddings from an HDF5 file."""
    with h5py.File(h5_file, 'r') as f:
        features = f['/features'][:]
    return features

def get_sample_name_from_filename(filename):
    """Extract core sample name from HDF5 filename."""
    parts = filename.split('-')[:4]
    return '-'.join(parts)

def process_batches(input_dir, metadata_file, output_pytorch, output_ludwig):
    """Iterate over batch directories and extract embeddings."""
    metadata_df = pd.read_csv(metadata_file)
    metadata_dict = metadata_df.set_index('sample_name')['label'].to_dict()
    
    pytorch_columns = ['sample_name', 'label'] + [f'feature_{i+1}' for i in range(1536)]
    pytorch_df = pd.DataFrame(columns=pytorch_columns)
    
    ludwig_columns = ['sample_name', 'label', 'embedding']
    ludwig_df = pd.DataFrame(columns=ludwig_columns)
    
    for batch_folder in sorted(os.listdir(input_dir)):
        if not batch_folder.startswith('batch_'):
            continue
        features_path = os.path.join(input_dir, batch_folder, '20x_256px_0px_overlap/features_uni_v2')
        if not os.path.exists(features_path):
            continue
        
        for h5_file in sorted(os.listdir(features_path)):
            if not h5_file.endswith('.h5'):
                continue
            
            h5_path = os.path.join(features_path, h5_file)
            sample_name = get_sample_name_from_filename(h5_file)
            label = metadata_dict.get(sample_name, 'Unknown')
            
            embeddings = extract_embeddings(h5_path)
            
            for vector in embeddings:
                pytorch_df = pd.DataFrame([[sample_name, label] + vector.tolist()], columns=pytorch_columns)
                pytorch_df.to_csv(output_pytorch, mode='a', header=not os.path.exists(output_pytorch), index=False)
                
                ludwig_vector = ' '.join(map(str, vector))
                ludwig_df = pd.DataFrame([[sample_name, label, ludwig_vector]], columns=ludwig_columns)
                ludwig_df.to_csv(output_ludwig, mode='a', header=not os.path.exists(output_ludwig), index=False)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract embeddings from HDF5 files.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing batch folders')
    parser.add_argument('--metadata_file', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--output_pytorch', type=str, required=True, help='Path to save PyTorch-style CSV')
    parser.add_argument('--output_ludwig', type=str, required=True, help='Path to save Ludwig-style CSV')
    args = parser.parse_args()
    
    process_batches(args.input_dir, args.metadata_file, args.output_pytorch, args.output_ludwig)

