import pandas as pd
import re
import sys

def merge_vector_columns(csv_file, new_csv_file):
    """
    Merges columns named vector1 to vector2048 in a CSV, creates an 'embedding' column,
    removes the vector columns, and saves the result to a new CSV file.

    Args:
        csv_file (str): Path to the original CSV file.
        new_csv_file (str): Path to save the new CSV file.
    """

    try:
        df = pd.read_csv(csv_file)

        # Identify vector columns
        vector_columns = [col for col in df.columns if re.match(r'vector\d+', col)]

        if not vector_columns:
            print(f"No columns named 'vector{{n}}' found in {csv_file}.")
            return

        # Merge vector columns into a single 'embedding' column
        df['embedding'] = df[vector_columns].astype(str).apply(lambda row: ' '.join(row), axis=1)

        # Remove the original vector columns
        df.drop(columns=vector_columns, inplace=True)

        # Save the modified DataFrame to the new CSV file
        df.to_csv(new_csv_file, index=False)

        print(f"Successfully merged vector columns. New file saved as: {new_csv_file}")

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_file.csv> <output_file.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    merge_vector_columns(input_file, output_file)
