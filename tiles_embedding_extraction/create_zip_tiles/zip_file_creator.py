import os
import zipfile
import time

def zip_image_tiles(root_dir, output_zip):
    """
    Zip PNG files from '_tiles' directories within 'TCGA' directories into a single zip file.

    Args:
        root_dir (str): Root directory containing 'TCGA' subdirectories.
        output_zip (str): Path to the output zip file.
    """
    start_time = time.time()
    print(f"Starting to zip PNG files from {root_dir} to {output_zip}", flush=True)

    total_files_added = 0

    # Open the zip file in 'w' mode to create a new zip file
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_STORED) as zipf:
        # Traverse the directory tree
        for dirpath, _, filenames in os.walk(root_dir):
            # Check if parent directory starts with 'TCGA' and current directory ends with '_tiles'
            parent_dir = os.path.basename(os.path.dirname(dirpath))
            current_dir = os.path.basename(dirpath)
            if parent_dir.startswith('TCGA') and current_dir.endswith('_tiles'):
                print(f"Processing {dirpath}", flush=True)
                png_files = [f for f in filenames if f.endswith('.png')]
                if not png_files:
                    print(f"No PNG files found in {dirpath}, skipping", flush=True)
                    continue

                # Add each PNG file to the zip
                for file in png_files:
                    file_path = os.path.join(dirpath, file)
                    rel_path = os.path.relpath(file_path, root_dir)
                    try:
                        zipf.write(file_path, rel_path)
                        total_files_added += 1
                        if total_files_added % 10000 == 0:
                            print(f"Added {total_files_added} files so far", flush=True)
                    except Exception as e:
                        print(f"Error adding {file_path}: {e}", flush=True)

                print(f"Added {len(png_files)} files from {dirpath}", flush=True)

    elapsed_time = time.time() - start_time
    print(f"Completed zipping {total_files_added} files to {output_zip} in {elapsed_time:.2f} seconds", flush=True)

    # Verify the zip file
    if os.path.exists(output_zip) and zipfile.is_zipfile(output_zip):
        with zipfile.ZipFile(output_zip, 'r') as zip_check:
            zipped_count = len(zip_check.namelist())
            print(f"Verified {zipped_count} entries in {output_zip}", flush=True)
            if zipped_count != total_files_added:
                print(f"Warning: Expected {total_files_added} files, but ZIP contains {zipped_count}", flush=True)
    else:
        print(f"Error: {output_zip} is not a valid ZIP file", flush=True)

if __name__ == "__main__":
    root_directory = os.path.abspath("/share/lab_goecks/TCGA_deep_learning/batch_lists/")
    output_zip_file = "image_tiles_full.zip"
    zip_image_tiles(root_directory, output_zip_file)
