import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from multiprocessing import Pool, cpu_count

def process_chunk(chunk_args):
    """Process a single chunk of the CSV file and return aggregated results."""
    chunk, usecols = chunk_args
    # Clean and process chunk
    chunk['sample_name'] = chunk['sample_name'].str.strip().str.lower()
    chunk = chunk.dropna(subset=['sample_name'])

    # Convert label to numeric, drop invalid rows
    chunk['label'] = pd.to_numeric(chunk['label'].str.strip(), errors='coerce')
    chunk = chunk.dropna(subset=['label']).astype({'label': int})

    # Aggregate results
    sample_counts = Counter(chunk['sample_name'])
    label_counts = Counter(chunk['label'])
    unique_samples = set(chunk['sample_name'])
    tiles_per_sample = chunk.groupby('sample_name')['label'].first()  # First label per sample

    return sample_counts, label_counts, unique_samples, tiles_per_sample.to_dict()

def analyze_csv(file_path: str, chunk_size: int = 100000):
    """Analyze and visualize data from the embeddings CSV file efficiently with parallel processing."""

    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Only load required columns
    usecols = ['sample_name', 'label']
    print(f"Reading CSV file '{file_path}' in chunks of {chunk_size} rows with {cpu_count()} workers...", flush=True)

    # Prepare for parallel processing
    chunk_iterator = pd.read_csv(
        file_path,
        usecols=usecols,
        dtype={'sample_name': str, 'label': str},
        encoding='utf-8',
        skipinitialspace=True,
        delimiter=',',
        chunksize=chunk_size
    )

    # Use multiprocessing Pool to process chunks in parallel
    total_rows = 0
    sample_counts = Counter()
    label_counts = Counter()
    unique_samples = set()
    tiles_per_sample_dict = {}

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_chunk, [(chunk, usecols) for chunk in chunk_iterator])

    # Combine results from all chunks
    for sc, lc, us, tps in results:
        sample_counts.update(sc)
        label_counts.update(lc)
        unique_samples.update(us)
        tiles_per_sample_dict.update(tps)  # Labels per sample
        total_rows += sum(sc.values())

    num_unique_samples = len(unique_samples)
    print(f"Number of rows: {total_rows:,}", flush=True)
    print(f"Number of unique sample names: {num_unique_samples:,}", flush=True)
    print(f"Count of label 0: {label_counts[0]:,}", flush=True)
    print(f"Count of label 1: {label_counts[1]:,}", flush=True)

    # Prepare output directory
    output_dir = "./analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Figure 1: Histogram of bag sizes (tiles per sample)
    print("Generating histogram of bag sizes...", flush=True)
    bag_sizes = list(sample_counts.values())  # Bag size is the number of tiles per sample
    plt.figure(figsize=(12, 6))
    plt.hist(bag_sizes, bins=50, log=False)  # Removed log scale for standard frequency view
    plt.title('Distribution of Bag Sizes')
    plt.xlabel('Bag Size (Number of Tiles per Sample)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    hist_plot_file = os.path.join(output_dir, f"bag_size_distribution_{timestamp}.png")
    plt.savefig(hist_plot_file)
    plt.close()
    print(f"Histogram saved to: {hist_plot_file}", flush=True)

    # Figure 2: Boxplot of tiles per sample by label (using full dataset)
    print("Generating boxplot of tiles per sample by label...", flush=True)
    tiles_per_sample = pd.DataFrame({
        'sample_name': list(sample_counts.keys()),
        'tile_count': list(sample_counts.values())
    })
    tiles_per_sample['label'] = tiles_per_sample['sample_name'].map(tiles_per_sample_dict)

    # Filter to valid labels (0 and 1)
    tiles_df = tiles_per_sample[tiles_per_sample['label'].isin([0, 1])]

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
