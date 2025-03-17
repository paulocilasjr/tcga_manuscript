import os
import zipfile
import time

def zip_image_tiles(root_dir, output_zip):
    """
    Zip PNG files from directories ending with '_tiles' sequentially.

    Args:
        root_dir (str): Root directory containing case subdirectories.
        output_zip (str): Path to the output zip file.
    """
    start_time = time.time()
    print(f"Starting to zip PNG files from {root_dir} to {output_zip}", flush=True)

    # Use ZIP_STORED for no compression
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_STORED) as zipf:
        png_files = []
        
        # Use os.walk for efficient file discovery
        print("Scanning directories for PNG files in '_tiles' subdirectories...", flush=True)
        for dirpath, _, filenames in os.walk(root_dir):
            if os.path.basename(dirpath).endswith('_tiles'):
                for file in filenames:
                    if file.endswith(".png"):
                        file_path = os.path.join(dirpath, file)
                        png_files.append(file_path)
        
        total_files = len(png_files)
        print(f"Found {total_files} PNG files in '_tiles' directories to zip", flush=True)
        
        if total_files == 0:
            print("No PNG files found in '_tiles' directories, exiting", flush=True)
            return

        # Sequential zipping with progress tracking
        print("Zipping files...", flush=True)
        for i, file_path in enumerate(png_files, 1):
            try:
                rel_path = os.path.relpath(file_path, root_dir)
                zipf.write(file_path, rel_path)
                if i % 10000 == 0:
                    print(f"Zipped {i}/{total_files} files", flush=True)
            except Exception as e:
                print(f"Error zipping {file_path}: {e}", flush=True)

    elapsed_time = time.time() - start_time
    print(f"Created zip file: {output_zip} with {total_files} files in {elapsed_time:.2f} seconds", flush=True)

    # Verify ZIP integrity
    if os.path.exists(output_zip) and zipfile.is_zipfile(output_zip):
        with zipfile.ZipFile(output_zip, 'r') as zip_check:
            zipped_count = len(zip_check.namelist())
            print(f"Verified {zipped_count} entries in {output_zip}", flush=True)
            if zipped_count != total_files:
                print(f"Warning: Expected {total_files} files, but ZIP contains {zipped_count}", flush=True)
    else:
        print(f"Error: {output_zip} is not a valid ZIP file", flush=True)

if __name__ == "__main__":
    root_directory = os.path.abspath("/share/lab_goecks/TCGA_deep_learning/batch_lists/")
    output_zip_file = "image_tiles_full.zip"
    zip_image_tiles(root_directory, output_zip_file)
