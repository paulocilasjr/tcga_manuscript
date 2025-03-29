import pandas as pd
import sys

def merge_vector_columns(input_file, output_file, chunksize=630000):
    first_chunk = True
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        # Identify 'vector' columns
        vector_cols = [col for col in chunk.columns if col.startswith('vector')]
        # Create 'embeddings' column
        chunk['embeddings'] = chunk[vector_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        # Select columns to keep
        columns_to_keep = [col for col in chunk.columns if not col.startswith('vector')]
        chunk_final = chunk[columns_to_keep + ['embeddings']]
        # Write to output file
        chunk_final.to_csv(output_file, mode='a', header=first_chunk, index=False)
        first_chunk = False
    print(f"Processed file saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file.csv output_file.csv")
        sys.exit(1)
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    merge_vector_columns(input_filename, output_filename, chunksize=630000)
