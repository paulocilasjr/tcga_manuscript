import pandas as pd
import argparse
import logging

# Configure logging to display warnings
logging.basicConfig(level=logging.WARNING)

def modify_sample_name(name: str) -> str:
    """
    Modify the sample_name by keeping everything up to and including '-01',
    but only if '-01' appears after the third '-'.
    
    Args:
        name: The original sample name string.

    Returns:
        The modified sample name.
    """
    parts = name.split("-")
    if len(parts) > 3:
        potential_split = "-".join(parts[:3]) + "-01"
        if name.startswith(potential_split):
            return potential_split
    
    logging.warning(f"Skipping modification: '{name}' does not meet split criteria.")
    return name

def main(input_file: str, output_file: str):
    """
    Read a CSV file, modify the 'sample_name' column, and save to a new CSV file.
    
    Args:
        input_file: Path to the input CSV file.
        output_file: Path to the output CSV file.
    """
    print(f"Processing {input_file}...")
    
    # Read the CSV file in chunks to optimize memory usage
    chunk_size = 10000  # Adjust as needed
    with pd.read_csv(input_file, chunksize=chunk_size) as reader:
        for chunk in reader:
            if "sample_name" not in chunk.columns:
                raise ValueError("CSV file must contain 'sample_name' column")
            
            # Modify the 'sample_name' column
            chunk["sample_name"] = chunk["sample_name"].map(modify_sample_name)
            
            # Append to the output file
            chunk.to_csv(output_file, mode='a', index=False, header=not pd.io.common.file_exists(output_file))
    
    print("Done.")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Modify 'sample_name' in a CSV file by removing everything after '-01', only if it appears after the third '-'")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to the output CSV file")
    args = parser.parse_args()
    
    # Run the main function with provided arguments
    main(args.input_file, args.output_file)
