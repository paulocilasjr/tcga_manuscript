import argparse
import pandas as pd
import logging
import os
from multiprocessing import cpu_count

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Add labels to embeddings CSV and optionally process in Ludwig format.')
parser.add_argument('--embedding_file', type=str, required=True, help='Path to the embeddings CSV file')
parser.add_argument('--metadata_file', type=str, required=True, help='Path to the metadata CSV file')
parser.add_argument('--output_file', type=str, required=True, help='Path to save the output CSV file')
parser.add_argument('--processing_type', type=str, choices=['ludwig_format', 'pytorch_format'], required=True,
                    help='Processing type: ludwig_format or pytorch_format')
parser.add_argument('--core_column_name', type=str, default=None,
                    help='Core column name to merge if ludwig_format (e.g., "vector_")')
parser.add_argument('--num_cores', type=int, default=min(4, cpu_count()), help='Number of CPU cores to use')
parser.add_argument('--chunk_size', type=int, default=100_000, help='Number of rows to process per chunk')
args = parser.parse_args()

# Validate core_column_name for ludwig_format
if args.processing_type == 'ludwig_format' and not args.core_column_name:
    parser.error('--core_column_name is required when --processing_type is ludwig_format')

def main():
    print(f"Starting processing with embedding file: {args.embedding_file}", flush=True)
    print(f"Metadata file: {args.metadata_file}", flush=True)
    print(f"Output file: {args.output_file}", flush=True)
    print(f"Processing type: {args.processing_type}", flush=True)
    if args.core_column_name:
        print(f"Core column name for merging: {args.core_column_name}", flush=True)
    print(f"Using {args.num_cores} CPU cores with chunk size {args.chunk_size}", flush=True)

    # Read metadata into dictionary
    print("Loading metadata...", flush=True)
    metadata_df = pd.read_csv(args.metadata_file, dtype={'label': 'int32'})
    metadata_dict = metadata_df.set_index('sample_name')['label'].to_dict()
    print(f"Metadata entries loaded: {len(metadata_dict)}", flush=True)

    # Determine dtypes for optimized reading
    print("Determining column dtypes...", flush=True)
    sample_df = pd.read_csv(args.embedding_file, nrows=1)
    first_col = sample_df.columns[0]  # Usually the sample name
    dtypes = {
        col: 'float32' if col != first_col else str
        for col in sample_df.columns
    }

    # Begin chunked processing
    first_chunk = True
    total_rows = 0
    unmatched_names = []

    print("Processing and writing chunks...", flush=True)
    for chunk in pd.read_csv(args.embedding_file, chunksize=args.chunk_size, dtype=dtypes):
        total_rows += len(chunk)

        # Add label column
        chunk['label'] = chunk[first_col].map(metadata_dict).astype('Int64')

        # Handle ludwig format
        if args.processing_type == 'ludwig_format' and args.core_column_name:
            vector_cols = [col for col in chunk.columns if col.startswith(args.core_column_name)]
            if vector_cols:
                chunk['merged_vector'] = chunk[vector_cols].astype(str).agg(' '.join, axis=1)
                chunk.drop(columns=vector_cols, inplace=True)

        # Track unmatched sample names (first few only)
        if chunk['label'].isna().any():
            unmatched = chunk[chunk['label'].isna()][first_col].tolist()
            unmatched_names.extend(unmatched[:5 - len(unmatched_names)])

        # Write chunk to output file
        chunk.to_csv(args.output_file, index=False, mode='w' if first_chunk else 'a', header=first_chunk)
        first_chunk = False

    print(f"Finished processing. Total rows processed: {total_rows}", flush=True)

    if unmatched_names:
        print(f"Unmatched sample names (up to 5 shown):", flush=True)
        for name in unmatched_names:
            print(f" - {name}", flush=True)
    else:
        print("All sample names matched successfully.", flush=True)

if __name__ == "__main__":
    main()

