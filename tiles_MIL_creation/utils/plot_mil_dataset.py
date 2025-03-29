import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime

def analyze_csv(file_path: str):
    """Analyze and visualize data from the bagged embeddings CSV file."""

    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Read CSV with proper encoding and space trimming
    print(f"Reading CSV file: {file_path}", flush=True)
    try:
        df = pd.read_csv(file_path, dtype={'bag_samples': str, 'bag_label': str, 'split': int}, encoding='utf-8', skipinitialspace=True, delimiter=',')
    except Exception as e:
        print(f"Error: Failed to read CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure required columns exist
    required_columns = ['bag_samples', 'bag_size', 'bag_label', 'split']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}", file=sys.stderr)
        sys.exit(1)

    # 🔹 Ensure 'bag_samples' has no leading/trailing spaces and is in correct format
    df['bag_samples'] = df['bag_samples'].astype(str).str.strip().str.lower()

    # Remove any NaN values in 'bag_samples'
    df = df.dropna(subset=['bag_samples'])

    # 🔹 Convert 'bag_label' column properly
    df['bag_label'] = df['bag_label'].astype(str).str.strip()
    df['bag_label'] = pd.to_numeric(df['bag_label'], errors='coerce')
    df = df.dropna(subset=['bag_label'])
    df['bag_label'] = df['bag_label'].astype(int)

    # Count number of rows
    num_rows = len(df)
    print(f"Number of rows: {num_rows}", flush=True)

    # Count occurrences of 0 and 1 in bag_label
    label_counts = df['bag_label'].value_counts().reindex([0, 1], fill_value=0)
    print(f"Count of label 0: {label_counts[0]}", flush=True)
    print(f"Count of label 1: {label_counts[1]}", flush=True)

    # Prepare output directory
    output_dir = "./analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Assign numeric labels to bag_samples
    df['bag_sample_id'] = pd.factorize(df['bag_samples'])[0] + 1

    # Figure 1: Histogram of bag sizes (discrete bins, shows all sizes including small ones)
    print("Generating histogram of bag sizes...", flush=True)
    plt.figure(figsize=(12, 6))

    # Calculate the full range of integer bag sizes
    min_size = int(df['bag_size'].min())
    max_size = int(df['bag_size'].max())
    bins = range(min_size, max_size + 2)  # +2 to include the last bin edge

    # Plot with clearly defined bins
    ax = sns.histplot(df['bag_size'], bins=bins, discrete=True, color='skyblue', edgecolor='black')

    # Annotate bars with frequency count
    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(str(height), (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8)

    plt.title('Distribution of Bag Sizes')
    plt.xlabel('Bag Size')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    hist_plot_file = os.path.join(output_dir, f"bag_size_distribution_{timestamp}.png")
    plt.savefig(hist_plot_file)
    plt.close()
    print(f"Histogram saved to: {hist_plot_file}", flush=True)

    # Figure 2: Bar plot of label counts per split
    print("Generating bar plot of label counts per split...", flush=True)
    split_label_counts = df.groupby(['split', 'bag_label']).size().unstack(fill_value=0)
    split_label_counts.plot(kind='bar', stacked=False, figsize=(10, 6))
    plt.title('Count of Labels (0 and 1) per Split')
    plt.xlabel('Split (0, 1, 2)')
    plt.ylabel('Total Count')
    plt.legend(title='Bag Label', labels=['Label 0', 'Label 1'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    split_bar_plot_file = os.path.join(output_dir, f"label_counts_per_split_{timestamp}.png")
    plt.savefig(split_bar_plot_file)
    plt.close()
    print(f"Bar plot saved to: {split_bar_plot_file}", flush=True)

def main():
    """Main function to run the analysis."""
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the CSV file as an argument.", file=sys.stderr)
        print("Usage: python analyze_bagged_data.py <csv_file>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_csv(file_path)

if __name__ == "__main__":
    main()

