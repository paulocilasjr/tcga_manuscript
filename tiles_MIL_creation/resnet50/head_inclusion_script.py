import pandas as pd

# Path to the input CSV file
csv_file = 'tcga_mil_resnet50_pytorch_by_sample_balanced.csv'

# Read the CSV file without headers
df = pd.read_csv(csv_file, header=None)

# Total number of columns
total_columns = df.shape[1]

# Generate headers
fixed_headers = ['bag_samples', 'bag_size', 'bag_label', 'split']
vector_headers = [f'vector{i}' for i in range(total_columns - len(fixed_headers))]

# Combine headers
all_headers = fixed_headers + vector_headers

# Assign headers to the DataFrame
df.columns = all_headers

# Save the DataFrame back to CSV (overwrite or create a new file)
df.to_csv(csv_file, index=False)

print(f"Headers added successfully to {csv_file}")
