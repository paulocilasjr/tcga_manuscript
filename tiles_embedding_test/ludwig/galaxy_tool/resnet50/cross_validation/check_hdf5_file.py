import h5py
import numpy as np

# Specify the path to your HDF5 file
file_path = '07471b9a039311f0ad25cc483a242c13.training.hdf5'  # Replace with your file's path

# Open the HDF5 file in read mode
with h5py.File(file_path, 'r') as f:
    # Print the top-level keys (groups or datasets)
    print("Top-level keys in the HDF5 file:")
    print(list(f.keys()))

    # Function to recursively print the structure
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}")
            print(f"  Shape: {obj.shape}")
            print(f"  Dtype: {obj.dtype}")
            # Optionally print a small sample of the data
            if obj.size < 100:  # Limit to small datasets for readability
                print(f"  Data: {obj[()]}")
            else:
                print(f"  Sample Data (first 5 elements): {obj[:5]}")

    # Visit all items in the file and print their structure
    print("\nFull structure of the HDF5 file:")
    f.visititems(print_structure)

    # Example: Access a specific dataset (replace 'dataset_name' with an actual name)
    # dataset = f['dataset_name']
    # print("\nAccessing a specific dataset:")
    # print(f"Data: {dataset[()]}")  # Use [()] to load the entire dataset into memory
