import argparse
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Add labels to embeddings CSV and update sample names.')
parser.add_argument('--embedding_file', type=str, required=True, help='Path to the embeddings CSV file')
parser.add_argument('--metadata_file', type=str, required=True, help='Path to the metadata CSV file')
parser.add_argument('--output_file', type=str, required=True, help='Path to save the output CSV file')
parser.add_argument('--processing_type', type=str, choices=['ludwig_format', 'pytorch_format'], required=True,
                    help='Processing type: ludwig_format or pytorch_format')
parser.add_argument('--core_column_name', type=str, default=None,
                    help='Core column name to merge if ludwig_format (e.g., "vector_")')
args = parser.parse_args()

# Validate core_column_name requirement for ludwig_format
if args.processing_type == 'ludwig_format' and not args.core_column_name:
    parser.error('--core_column_name is required when --processing_type is ludwig_format')

print(f"Starting processing with embedding file: {args.embedding_file}", flush=True)
print(f"Metadata file: {args.metadata_file}", flush=True)
print(f"Output file will be saved to: {args.output_file}", flush=True)
print(f"Processing type: {args.processing_type}", flush=True)
if args.core_column_name:
    print(f"Core column name for merging: {args.core_column_name}", flush=True)

# Read the files
print("Reading embeddings CSV...", flush=True)
embeddings_df = pd.read_csv(args.embedding_file)
print(f"Embeddings CSV loaded with {len(embeddings_df)} rows and {len(embeddings_df.columns)} columns", flush=True)

print("Reading metadata CSV...", flush=True)
metadata_df = pd.read_csv(args.metadata_file)
print(f"Metadata CSV loaded with {len(metadata_df)} rows and {len(metadata_df.columns)} columns", flush=True)

# Store the original number of rows for verification
original_row_count = len(embeddings_df)
print(f"Original row count in embeddings file: {original_row_count}", flush=True)

# Create a dictionary mapping sample_name from metadata to label
print("Creating metadata dictionary...", flush=True)
metadata_dict = metadata_df.set_index('sample_name')['label'].to_dict()
print(f"Metadata dictionary created with {len(metadata_dict)} entries", flush=True)

# Function to find the matching sample name from metadata and replace it
def find_matching_sample_name(embedding_sample_name):
    for metadata_sample_name in metadata_dict.keys():
        if metadata_sample_name in embedding_sample_name:
            return metadata_sample_name
    return embedding_sample_name  # Keep original name if no match found

# Update sample_name and add labels
print("Updating sample names in embeddings dataframe...", flush=True)
original_sample_col = embeddings_df.iloc[:, 0].copy()  # Store original sample names
embeddings_df.iloc[:, 0] = embeddings_df.iloc[:, 0].apply(find_matching_sample_name)
print("Adding labels to embeddings dataframe...", flush=True)
embeddings_df['label'] = embeddings_df.iloc[:, 0].map(metadata_dict)  # NaN will be assigned if no match

# Identify and print unmatched sample names
unmatched = embeddings_df[embeddings_df['label'].isna()][embeddings_df.columns[0]].tolist()
if unmatched:
    print("No matches found for the following sample names:", flush=True)
    for name in unmatched:
        print(f" - {name}", flush=True)
    print(f"Total unmatched sample names: {len(unmatched)}", flush=True)
else:
    print("All sample names matched successfully.", flush=True)

# If ludwig_format, merge columns with core_column_name into a single string
if args.processing_type == 'ludwig_format':
    print(f"Processing in ludwig_format, looking for columns starting with '{args.core_column_name}'...", flush=True)
    # Find columns that start with core_column_name
    core_cols = [col for col in embeddings_df.columns if col.startswith(args.core_column_name)]
    if not core_cols:
        print(f"Warning: No columns found starting with '{args.core_column_name}'", flush=True)
    else:
        print(f"Found {len(core_cols)} columns to merge: {core_cols}", flush=True)
        # Merge the columns into a single whitespace-separated string
        embeddings_df['merged_vector'] = embeddings_df[core_cols].astype(str).agg(' '.join, axis=1)
        print("Merged columns into 'merged_vector'", flush=True)
        # Drop the original vector columns if desired (optional)
        embeddings_df = embeddings_df.drop(columns=core_cols)
        print(f"Dropped {len(core_cols)} original vector columns", flush=True)

# Save the updated file
print("Saving updated dataframe to CSV...", flush=True)
embeddings_df.to_csv(args.output_file, index=False)
print(f"Output saved to {args.output_file}", flush=True)

# Verify the number of lines in input and output files
print("Verifying row count in output file...", flush=True)
output_df = pd.read_csv(args.output_file)
output_row_count = len(output_df)
print(f"Output file row count: {output_row_count}", flush=True)

if original_row_count == output_row_count:
    print(f"Verification passed: Input file ({original_row_count} rows) matches output file ({output_row_count} rows).", flush=True)
else:
    print(f"Verification failed: Input file has {original_row_count} rows, but output file has {output_row_count} rows.", flush=True)
    print("Possible data loss detected. Check the output file for missing rows.", flush=True)
