import csv
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import multiprocessing as mp
import argparse
import uuid
import tempfile

# Configure logging
logging.basicConfig(
    filename="/tmp/ludwig_embeddings.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)

def parse_bag_size(value):
    if "-" in value:
        min_val, max_val = map(int, value.split("-"))
        if min_val > max_val:
            raise argparse.ArgumentTypeError("Invalid range: min value cannot be greater than max value.")
        return [min_val, max_val]
    return [int(value), int(value)]

def parse_by_sample(value):
    try:
        value = str(value)
        splits = [int(x) for x in value.split(",")]
        valid_splits = {0, 1, 2}
        if not all(x in valid_splits for x in splits):
            logging.warning(f"Invalid splits in by_sample: {splits}. Defaulting to random/balanced bagging.")
            return None
        return splits
    except (ValueError, AttributeError):
        logging.warning(f"Could not parse by_sample value: {value}. Defaulting to random/balanced bagging.")
        return None

def load_metadata(file_path):
    metadata = pd.read_csv(file_path)
    print(f"Metadata loaded with {len(metadata)} samples. Columns: {list(metadata.columns)}", flush=True)
    print(f"Unique sample_names: {metadata['sample_name'].nunique()}, Unique labels: {metadata['label'].nunique()}", flush=True)
    return metadata

def str_array_split(split_proportions):
    split_array = [float(p) for p in split_proportions.split(",")]
    if len(split_array) == 2:
        split_array.insert(1, 0.0)
    return split_array

def split_sizes(num_samples, proportions):
    sizes = [int(p * num_samples) for p in proportions]
    sizes[-1] = num_samples - sum(sizes[:-1])
    return sizes

def split_data(metadata, split_proportions, dataleak=False):
    proportions = str_array_split(split_proportions)
    if dataleak:
        list_samples = metadata["sample_name"].unique()
    else:
        list_samples = metadata["sample_name"].values

    num_samples = len(list_samples)
    sizes = split_sizes(num_samples, proportions)

    shuffled_samples = np.random.permutation(list_samples)

    split_values = np.zeros(num_samples, dtype=int)
    if sizes[1] > 0:
        split_values[sizes[0]:sizes[0] + sizes[1]] = 1
    split_values[sizes[0] + sizes[1]:] = 2

    split_series = pd.Series(split_values, index=shuffled_samples)
    metadata["split"] = metadata["sample_name"].map(split_series)
    train_size = sum(metadata["split"] == 0)
    val_size = sum(metadata["split"] == 1)
    test_size = sum(metadata["split"] == 2)
    print(f"Data split into train: {train_size}, val: {val_size}, test: {test_size} samples.", flush=True)
    return metadata

def aggregate_embeddings(embeddings, pooling_method, use_gpu=False):
    if pooling_method == "max_pooling":
        return np.max(embeddings, axis=0)
    if pooling_method == "mean_pooling":
        return np.mean(embeddings, axis=0)
    if pooling_method == "sum_pooling":
        return np.sum(embeddings, axis=0)
    if pooling_method == "min_pooling":
        return np.min(embeddings, axis=0)
    if pooling_method == "median_pooling":
        return np.median(embeddings, axis=0)
    if pooling_method == "l2_norm_pooling":
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.mean(embeddings / (norm + 1e-8), axis=0) if norm.any() else np.mean(embeddings, axis=0)
    if pooling_method == "geometric_mean_pooling":
        return np.exp(np.mean(np.log(np.clip(embeddings, 1e-10, None)), axis=0))
    if pooling_method == "first_embedding":
        return embeddings[0]
    if pooling_method == "last_embedding":
        return embeddings[-1]
    if pooling_method == "attention_pooling":
        device = 'cuda' if use_gpu else 'cpu'
        tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        weights = nn.Softmax(dim=0)(nn.Linear(tensor.shape[1], 1).to(device)(tensor))
        return torch.sum(weights * tensor, dim=0).cpu().numpy()
    # Add gated_pooling if needed

def bag_by_sample_process(group, pooling_method, bag_size, use_gpu):
    embeddings = group[group.columns.difference(['sample_name', 'label', 'split'])].values
    sample_names = group["sample_name"].values
    labels = group["label"].values
    split = group["split"].iloc[0]

    num_instances = len(group)
    random_bag_size = np.random.randint(bag_size[0], bag_size[1] + 1)
    num_bags = (num_instances + random_bag_size - 1) // random_bag_size
    bags = []

    print(f"Sample {group['sample_name'].iloc[0]}: {num_instances} instances, creating {num_bags} bags", flush=True)
    for i in range(num_bags):
        start_idx = i * random_bag_size
        end_idx = min(start_idx + random_bag_size, num_instances)
        bag_embeddings = embeddings[start_idx:end_idx]
        bag_sample_names = sample_names[start_idx:end_idx]
        bag_labels = labels[start_idx:end_idx]
        aggregated_embeddings = aggregate_embeddings(bag_embeddings, pooling_method, use_gpu)
        bag_label = int(any(bag_labels == 1))
        bags.append({
            "bag_label": bag_label,
            "split": split,
            "bag_size": len(bag_sample_names),
            "bag_samples": list(bag_sample_names),
            "embedding": aggregated_embeddings
        })
    print(f"Created {len(bags)} bags for sample: {group['sample_name'].iloc[0]}", flush=True)
    return bags

def bag_turns_process(df_np, bag_sizes, pooling_method, use_gpu, target_label, target_count):
    indices_0 = np.where(df_np[:, 1] == 0)[0]
    indices_1 = np.where(df_np[:, 1] == 1)[0]
    np.random.shuffle(indices_0)
    np.random.shuffle(indices_1)

    make_bag_1 = True if target_label is None else (target_label == 1)
    bags = []
    bag_set = set()
    bag_count = 0

    while (len(indices_0) > 0 or len(indices_1) > 0) and (target_count is None or bag_count < target_count):
        bag_size = np.random.randint(bag_sizes[0], bag_sizes[1] + 1)
        num_1_samples = min(np.random.randint(1, bag_size + 1), len(indices_1)) if make_bag_1 and len(indices_1) > 0 else 0
        selected_indices_1 = indices_1[:num_1_samples]
        indices_1 = indices_1[num_1_samples:]
        num_0_samples = min(bag_size - len(selected_indices_1), len(indices_0))
        selected_indices_0 = indices_0[:num_0_samples]
        indices_0 = indices_0[num_0_samples:]
        bag_indices = np.concatenate([selected_indices_0, selected_indices_1])
        bag_data = df_np[bag_indices]

        if len(bag_data) < bag_size and len(indices_1) > 0:
            num_extra = min(bag_size - len(bag_data), len(indices_1))
            extra_indices = indices_1[:num_extra]
            indices_1 = indices_1[num_extra:]
            bag_data = np.vstack([bag_data, df_np[extra_indices]])

        make_bag_1 = (target_label == 1) if target_label is not None else not make_bag_1
        if len(bag_data) > 0:
            only_embeddings = bag_data[:, 3:]
            aggregated_embedding = aggregate_embeddings(only_embeddings, pooling_method, use_gpu)
            bag_label = int(any(bag_data[:, 1] == 1))
            bag_key = (tuple(map(tuple, only_embeddings)), len(bag_data), tuple(bag_data[:, 0]))
            if bag_key not in bag_set:
                bag_set.add(bag_key)
                bags.append({
                    "bag_label": bag_label,
                    "split": bag_data[0, 2],
                    "bag_size": len(bag_data),
                    "bag_samples": list(bag_data[:, 0]),
                    "embedding": aggregated_embedding
                })
                bag_count += 1
    print(f"Created {len(bags)} bags for split: {df_np[0, 2]}", flush=True)
    return bags

def bag_random_process(df_np, bag_sizes, pooling_method, use_gpu, target_label, target_count):
    np.random.shuffle(df_np)
    idx = 0
    bag_set = set()
    bags = []
    bag_count = 0
    while idx < len(df_np) and (target_count is None or bag_count < target_count):
        bag_size = np.random.randint(bag_sizes[0], bag_sizes[1] + 1)
        end_idx = min(idx + bag_size, len(df_np))
        bag_data = df_np[idx:end_idx]
        only_embeddings = bag_data[:, 3:]
        aggregated_embedding = aggregate_embeddings(only_embeddings, pooling_method, use_gpu)
        bag_label = int(any(bag_data[:, 1] == 1))
        if target_label is not None and bag_label != target_label:
            idx = end_idx
            continue
        bag_key = (tuple(map(tuple, only_embeddings)), len(bag_data), tuple(bag_data[:, 0]))
        if bag_key not in bag_set:
            bag_set.add(bag_key)
            bags.append({
                "bag_label": bag_label,
                "split": bag_data[0, 2],
                "bag_size": len(bag_data),
                "bag_samples": list(bag_data[:, 0]),
                "embedding": aggregated_embedding
            })
            bag_count += 1
        idx = end_idx
    print(f"Created {len(bags)} bags for split: {df_np[0, 2]}", flush=True)
    return bags

def write_csv(output_csv, list_embeddings, chunk_size=10000, append=False, ludwig_format=False):
    mode = "a" if append and os.path.exists(output_csv) else "w"
    with open(output_csv, mode=mode, encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
        headers = ["bag_samples", "bag_size", "bag_label", "split"]

        if not list_embeddings:
            if mode == "w":
                csv_writer.writerow(headers)
            print("No bags generated, writing empty CSV", flush=True)
            logging.info("No valid data found to write to CSV.")
            return

        first_item = list_embeddings[0]
        if ludwig_format:
            headers.append("embedding")
        elif isinstance(first_item["embedding"], np.ndarray):
            headers.extend([f"vector{i+1}" for i in range(len(first_item["embedding"]))])
        else:
            raise ValueError("Expected NumPy array for embedding when not using ludwig_format.")

        if mode == "w":
            csv_writer.writerow(headers)

        for i in range(0, len(list_embeddings), chunk_size):
            chunk = list_embeddings[i:i + chunk_size]
            for bag in chunk:
                row = [",".join(map(str, bag["bag_samples"])), bag["bag_size"], bag["bag_label"], bag["split"]]
                if ludwig_format:
                    if isinstance(bag["embedding"], np.ndarray):
                        embedding_str = " ".join(f"{x:.6f}" for x in bag["embedding"])
                    else:
                        raise ValueError("Embedding must be a NumPy array for ludwig_format.")
                    row.append(embedding_str)
                else:
                    if isinstance(bag["embedding"], np.ndarray):
                        row.extend(bag["embedding"].tolist())
                    else:
                        raise ValueError("Embedding must be a NumPy array when not using ludwig_format.")
                csv_writer.writerow(row)
            print(f"Wrote {len(chunk)} bags to {output_csv}", flush=True)

def balance_bags_split(split_bags, imbalance_cap, split_metadata, bag_sizes, pooling_method, balance_enforced, use_gpu, embeddings_path):
    bags_0 = [bag for bag in split_bags if bag["bag_label"] == 0]
    bags_1 = [bag for bag in split_bags if bag["bag_label"] == 1]
    num_bags_0 = len(bags_0)
    num_bags_1 = len(bags_1)

    min_count = min(num_bags_0, num_bags_1)
    imbalance = abs(num_bags_0 - num_bags_1) / min_count * 100 if min_count > 0 else float('inf')
    print(f"Initial imbalance for split: {imbalance:.2f}% (0: {num_bags_0}, 1: {num_bags_1})", flush=True)

    if imbalance <= imbalance_cap or min_count == 0:
        return split_bags

    majority_count = max(num_bags_0, num_bags_1)
    minority_label = 1 if num_bags_0 > num_bags_1 else 0
    minority_count = min(num_bags_0, num_bags_1)
    target_minority_count = int(majority_count * (1 / (1 + imbalance_cap / 100)))
    if target_minority_count < minority_count:
        target_minority_count = minority_count
    bags_needed = target_minority_count - minority_count

    print(f"Target minority count: {target_minority_count}, bags needed: {bags_needed}", flush=True)

    df_np = pd.merge(split_metadata, pd.read_csv(embeddings_path), on='sample_name').to_numpy()
    all_bags = split_bags.copy()
    remaining_bags_needed = bags_needed

    max_iterations = 10
    for iteration in range(max_iterations):
        if remaining_bags_needed <= 0:
            break
        print(f"Iteration {iteration + 1}: Need {remaining_bags_needed} more bags of label {minority_label}", flush=True)
        np.random.shuffle(df_np)
        if balance_enforced:
            extra_bags = bag_turns_process(df_np, bag_sizes, pooling_method, use_gpu, minority_label, remaining_bags_needed)
        else:
            extra_bags = bag_random_process(df_np, bag_sizes, pooling_method, use_gpu, minority_label, remaining_bags_needed)
        all_bags.extend(extra_bags)
        bags_0 = [bag for bag in all_bags if bag["bag_label"] == 0]
        bags_1 = [bag for bag in all_bags if bag["bag_label"] == 1]
        minority_count = len(bags_1) if minority_label == 1 else len(bags_0)
        remaining_bags_needed = target_minority_count - minority_count
        current_imbalance = abs(len(bags_0) - len(bags_1)) / min(len(bags_0), len(bags_1)) * 100 if min(len(bags_0), len(bags_1)) > 0 else float('inf')
        print(f"After iteration {iteration + 1}: Imbalance = {current_imbalance:.2f}% (0: {len(bags_0)}, 1: {len(bags_1)})", flush=True)

    final_imbalance = abs(len(bags_0) - len(bags_1)) / min(len(bags_0), len(bags_1)) * 100 if min(len(bags_0), len(bags_1)) > 0 else float('inf')
    print(f"Final imbalance for split: {final_imbalance:.2f}% (0: {len(bags_0)}, 1: {len(bags_1)})", flush=True)

    return all_bags

def truncate_bags(split_bags):
    bags_0 = [bag for bag in split_bags if bag["bag_label"] == 0]
    bags_1 = [bag for bag in split_bags if bag["bag_label"] == 1]
    minority_count = min(len(bags_0), len(bags_1))
    return bags_0[:minority_count] + bags_1[:minority_count]

def transform_bags_for_ludwig(bags):
    return [dict(bag, embedding=" ".join(map(str, bag["embedding"]))) for bag in bags]

# --- Map Step: Process one split using its temporary file ---
def process_split(split, metadata, pooling_method, balance_enforced, bag_sizes, use_gpu, split_temp_files, by_sample, embeddings_path, imbalance_cap, do_truncate_bags):
    print(f"Processing split: {split}", flush=True)
    split_metadata = metadata[metadata['split'] == split]
    split_bags = []

    split_temp_file = split_temp_files.get(split)
    if not split_temp_file or not os.path.exists(split_temp_file):
        print(f"No temp file found for split {split}", flush=True)
        return split_bags

    # Read the temporary file in chunks to reduce memory footprint.
    reader = pd.read_csv(split_temp_file, chunksize=5000)
    split_embeddings = pd.concat([chunk for chunk in reader])
    print(f"Split {split} embeddings rows: {len(split_embeddings)}", flush=True)

    if by_sample is not None and split in by_sample:
        print(f"Processing split {split} by sample", flush=True)
        for sample_name in split_metadata['sample_name'].unique():
            sample_embeddings = split_embeddings[split_embeddings['sample_name'] == sample_name]
            if not sample_embeddings.empty:
                sample_df = pd.merge(split_metadata[split_metadata['sample_name'] == sample_name], sample_embeddings, on='sample_name')
                print(f"Split {split}, sample {sample_name}: {len(sample_df)} rows after merge", flush=True)
                if not sample_df.empty:
                    bags = bag_by_sample_process(sample_df, pooling_method, bag_sizes, use_gpu)
                    split_bags.extend(bags)
            else:
                print(f"No embeddings found for sample {sample_name} in split {split}", flush=True)
    else:
        print(f"Processing split {split} with random/turns bagging", flush=True)
        split_df = pd.merge(split_metadata, split_embeddings, on='sample_name')
        print(f"Split {split}: {len(split_df)} rows after merge", flush=True)
        if not split_df.empty:
            df_np = split_df.to_numpy()
            if balance_enforced:
                bags = bag_turns_process(df_np, bag_sizes, pooling_method, use_gpu, None, None)
            else:
                bags = bag_random_process(df_np, bag_sizes, pooling_method, use_gpu, None, None)
            split_bags.extend(bags)

    if split_bags:
        if imbalance_cap is not None and split in metadata['split'].unique():
            split_bags = balance_bags_split(split_bags, imbalance_cap, split_metadata, bag_sizes, pooling_method, balance_enforced, use_gpu, embeddings_path)
            print(f"Balanced bags for split: {split}, now have {len(split_bags)} bags", flush=True)
        if do_truncate_bags:
            split_bags = truncate_bags(split_bags)
            print(f"Truncated bags for split: {split}, now have {len(split_bags)} bags", flush=True)

    print(f"Finished processing split: {split} with {len(split_bags)} bags", flush=True)
    return split_bags

# --- Main Bag Processing: Map & Reduce ---
def bag_processing(embeddings_path, metadata, pooling_method, balance_enforced, bag_sizes, repeats, ludwig_format, by_sample, use_gpu, output_csv, imbalance_cap, do_truncate_bags):
    bag_sizes = parse_bag_size(bag_sizes)
    required_cols = {"sample_name", "label"}
    if not required_cols.issubset(metadata.columns):
        missing = required_cols - set(metadata.columns)
        raise ValueError(f"Metadata CSV missing required columns: {missing}")

    sample_to_split = dict(zip(metadata['sample_name'], metadata['split']))
    splits = metadata['split'].unique()
    print(f"Starting processing for {len(splits)} splits: {list(splits)}", flush=True)

    by_sample = set(by_sample) if by_sample is not None else set()

    # Create temporary files for each split using tempfile
    split_temp_files = {}
    for split in splits:
        temp_fd, temp_path = tempfile.mkstemp(prefix=f"split_{split}_", suffix=".csv", dir=os.getcwd())
        os.close(temp_fd)
        split_temp_files[split] = temp_path

    # --- Map: Distribute embeddings CSV into temporary files using chunked reading ---
    buffer_size = 1000
    buffers = {temp_file: [] for temp_file in split_temp_files.values()}
    print(f"Distributing embeddings from {embeddings_path} to temporary files.", flush=True)
    for chunk in pd.read_csv(embeddings_path, chunksize=10000):
        for _, row in chunk.iterrows():
            sample_name = row['sample_name']
            if sample_name in sample_to_split:
                split = sample_to_split[sample_name]
                temp_file = split_temp_files.get(split)
                if temp_file:
                    buffers[temp_file].append(row.tolist())
                    if len(buffers[temp_file]) >= buffer_size:
                        with open(temp_file, 'a', newline='') as tf:
                            writer = csv.writer(tf)
                            # Write header if file is empty.
                            if os.stat(temp_file).st_size == 0:
                                writer.writerow(chunk.columns.tolist())
                            writer.writerows(buffers[temp_file])
                        buffers[temp_file] = []
    # Flush remaining buffers.
    for temp_file, rows in buffers.items():
        if rows:
            with open(temp_file, 'a', newline='') as tf:
                writer = csv.writer(tf)
                if os.stat(temp_file).st_size == 0:
                    # Using columns from the first chunk read earlier.
                    writer.writerow(chunk.columns.tolist())
                writer.writerows(rows)

    # --- Map: Process each split in parallel ---
    with mp.Pool(processes=mp.cpu_count()) as pool:
        args = [
            (split, metadata, pooling_method, balance_enforced, bag_sizes,
             use_gpu, split_temp_files, by_sample, embeddings_path, imbalance_cap, do_truncate_bags)
            for split in splits
        ]
        map_results = pool.starmap(process_split, args)

    # --- Reduce: Combine results from all splits ---
    all_bags = [bag for result in map_results for bag in result]
    print(f"Collected {len(all_bags)} total bags from all splits", flush=True)

    if not all_bags:
        print("No bags to write to CSV", flush=True)
    else:
        if ludwig_format:
            all_bags = transform_bags_for_ludwig(all_bags)
        write_csv(output_csv, all_bags, append=False, ludwig_format=ludwig_format)
        print(f"Completed bagging process and wrote results to {output_csv}", flush=True)

    # Centralized cleanup of temporary files
    for temp_file in split_temp_files.values():
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}", flush=True)
            except Exception as e:
                print(f"Error removing temporary file {temp_file}: {e}", flush=True)
                logging.error(f"Error removing temporary file {temp_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bags from embeddings and metadata")
    parser.add_argument("--embeddings_csv", type=str, required=True, help="Path to embeddings CSV")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--split_proportions", type=str, default='0.7,0.1,0.2', help="Proportions for train, val, test splits")
    parser.add_argument("--dataleak", action="store_true", help="Prevents data leakage when splitting")
    parser.add_argument("--balance_enforced", action="store_true", help="Enforce balanced bagging")
    parser.add_argument("--bag_size", type=str, required=True, help="Bag size (e.g., '4' or '3-5')")
    parser.add_argument("--pooling_method", type=str, required=True, help="Pooling method")
    parser.add_argument("--by_sample", type=parse_by_sample, default=None, help="Splits to bag by sample")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat bagging")
    parser.add_argument("--ludwig_format", action="store_true", help="Output in Ludwig-compatible format")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--imbalance_cap", type=int, default=None, help="Max percentage imbalance between positive and negative bags")
    parser.add_argument("--truncate_bags", action="store_true", help="Truncate excess bags to match minority label count")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="Use GPU for pooling operations")
    args = parser.parse_args()
    print(f"Starting bagging process with arguments: {args}", flush=True)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {args.random_seed}", flush=True)

    metadata_csv = load_metadata(args.metadata_csv)
    if "split" not in metadata_csv.columns:
        metadata_csv = split_data(metadata_csv, split_proportions=args.split_proportions, dataleak=args.dataleak)

    bag_processing(
        embeddings_path=args.embeddings_csv,
        metadata=metadata_csv,
        pooling_method=args.pooling_method,
        balance_enforced=args.balance_enforced,
        bag_sizes=args.bag_size,
        repeats=args.repeats,
        ludwig_format=args.ludwig_format,
        by_sample=args.by_sample,
        use_gpu=args.use_gpu,
        output_csv=args.output_csv,
        imbalance_cap=args.imbalance_cap,
        do_truncate_bags=args.truncate_bags
    )

