import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime

def analyze_csv(file_path: str):
    """Analyze and visualize data from the embeddings CSV file."""

    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Read CSV with proper encoding and space trimming
    print(f"Reading CSV file: {file_path}", flush=True)
    try:
        df = pd.read_csv(file_path, dtype={'sample_name': str, 'label': str}, encoding='utf-8', skipinitialspace=True, delimiter=',')
    except Exception as e:
        print(f"Error: Failed to read CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure required columns exist
    required_columns = ['sample_name', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}", file=sys.stderr)
        sys.exit(1)

    # 🔹 FIX: Ensure 'sample_name' has no leading/trailing spaces and is in correct format
    df['sample_name'] = df['sample_name'].astype(str).str.strip()

    # Remove any NaN values in the 'sample_name' column before counting unique samples
    rows_removed = df[df['sample_name'].isna()]  # Capture rows with NaN in 'sample_name'
    if not rows_removed.empty:
        print(f"DEBUG: Rows removed due to NaN 'sample_name':")
        print(rows_removed)  # Print the rows that are being removed for debugging
    df = df.dropna(subset=['sample_name'])

    # Optional: Normalize case sensitivity
    df['sample_name'] = df['sample_name'].str.lower()

    # 🔹 FIX: Convert 'label' column properly
    df['label'] = df['label'].astype(str).str.strip()  # Remove spaces
    df['label'] = pd.to_numeric(df['label'], errors='coerce')  # Convert to numeric (invalid -> NaN)
    df = df.dropna(subset=['label'])  # Remove rows where label is NaN
    df['label'] = df['label'].astype(int)  # Convert to integer

    # Ensure all unique sample names are counted correctly
    num_rows = len(df)
    num_unique_samples = len(df['sample_name'].unique())
    label_counts = df['label'].value_counts().reindex([0, 1], fill_value=0)  # Ensure both 0 and 1 are counted

    print(f"DEBUG: Unique sample names: {num_unique_samples}", flush=True)  # Check final count
    print(f"Number of rows: {num_rows}", flush=True)
    print(f"Number of unique sample names: {num_unique_samples}", flush=True)
    print(f"Count of label 0: {label_counts[0]}", flush=True)
    print(f"Count of label 1: {label_counts[1]}", flush=True)

    # Prepare output directory
    output_dir = "./analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Figure 1: Bar plot of number of rows per sample
    print("Generating bar plot of rows per sample...", flush=True)
    sample_counts = df['sample_name'].value_counts()
    plt.figure(figsize=(12, 6))
    sample_counts.plot(kind='bar')
    plt.title('Number of Rows per Sample')
    plt.xlabel('Sample Name')
    plt.ylabel('Number of Rows')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    bar_plot_file = os.path.join(output_dir, f"rows_per_sample_{timestamp}.png")
    plt.savefig(bar_plot_file)
    plt.close()
    print(f"Bar plot saved to: {bar_plot_file}", flush=True)

    # Figure 2: Boxplot of total tiles per sample by label
    print("Generating boxplot of tiles per sample by label...", flush=True)

    # Count tiles (rows) per sample and merge with labels
    tiles_per_sample = df.groupby('sample_name').size().reset_index(name='tile_count')
    sample_labels = df[['sample_name', 'label']].drop_duplicates()
    tiles_df = pd.merge(tiles_per_sample, sample_labels, on='sample_name', how='left')

    # Filter to only 0 and 1 labels (in case there are NaNs or unexpected values)
    tiles_df = tiles_df[tiles_df['label'].isin([0, 1])]

    if tiles_df.empty:
        print("Warning: No samples with labels 0 or 1 for boxplot.", file=sys.stderr)
    else:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='label', y='tile_count', data=tiles_df)
        plt.title('Distribution of Tiles per Sample by Label')
        plt.xlabel('Label (0 or 1)')
        plt.ylabel('Total Number of Tiles per Sample')
        plt.grid(True, linestyle='--', alpha=0.7)
        boxplot_file = os.path.join(output_dir, f"tiles_per_sample_boxplot_{timestamp}.png")
        plt.savefig(boxplot_file)
        plt.close()
        print(f"Boxplot saved to: {boxplot_file}", flush=True)

def main():
    """Main function to run the analysis."""
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the CSV file as an argument.", file=sys.stderr)
        print("Usage: python analyze_embeddings_output.py <csv_file>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_csv(file_path)

if __name__ == "__main__":
    main()

