import argparse
import csv
import logging
import multiprocessing as mp
import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# LOAD DATA FUNCTIONS
def load_metadata(file_path):
    """Loads metadata from a CSV file."""
    metadata = pd.read_csv(file_path)
    logging.info("Metadata loaded with %d samples, cols: %s",
                 len(metadata), list(metadata.columns))
    logging.info("Unique samples: %d, labels: %d",
                 metadata["sample_name"].nunique(),
                 metadata["label"].nunique())
    return metadata

# SPLIT FUNCTIONS
def str_array_split(split_proportions):
    """Converts a string of split proportions to a list of floats."""
    split_array = [float(p) for p in split_proportions.split(",")]
    if len(split_array) == 2:
        split_array.insert(1, 0.0)  # Insert validation proportion as 0 if only train and test are provided
    elif len(split_array) != 3:
        raise ValueError("split_proportions must have 2 or 3 values")
    return split_array


def split_sizes(num_samples, proportions):
    """Calculates split sizes based on proportions, ensuring they sum to the total number of samples."""
    sizes = [int(p * num_samples) for p in proportions]
    total = sum(sizes)
    if total < num_samples:
        sizes[-1] += num_samples - total  # Add remaining samples to test
    elif total > num_samples:
        sizes[0] -= total - num_samples  # Remove excess from train
    return sizes


def split_data(metadata, split_proportions, dataleak=False, random_seed=None):
    """
    Splits data into train, validation, and test sets based on provided proportions.

    Parameters:
    - metadata: DataFrame containing sample information with a 'sample_name' column.
    - split_proportions: String like '0.7,0.0,0.3' specifying train, validation, and test proportions.
    - dataleak: Boolean flag (unused here, included for compatibility).
    - random_seed: Optional integer to set random seed for reproducibility.

    Returns:
    - metadata: DataFrame with an added 'split' column (0 = train, 1 = validation, 2 = test).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Parse the split proportions
    proportions = str_array_split(split_proportions)
    train_prop, val_prop, test_prop = proportions

    # Get unique samples
    list_samples = metadata["sample_name"].unique()
    num_samples = len(list_samples)

    # Determine split assignments
    if val_prop == 0 and test_prop == 0:
        # Case 1: Both validation and test are 0, assign all to train
        split_values = np.zeros(num_samples, dtype=int)
    elif val_prop == 0:
        # Case 2: Validation is 0, split between train and test
        train_size = int(train_prop * num_samples)
        test_size = num_samples - train_size
        split_values = np.array([0] * train_size + [2] * test_size)
    else:
        # Case 3: Standard three-way split
        sizes = split_sizes(num_samples, proportions)
        split_values = np.concatenate([
            np.zeros(sizes[0], dtype=int),      # train
            np.ones(sizes[1], dtype=int),       # validation
            2 * np.ones(sizes[2], dtype=int)    # test
        ])

    # Shuffle and assign splits to samples
    shuffled_samples = np.random.permutation(list_samples)
    split_series = pd.Series(split_values, index=shuffled_samples)
    metadata["split"] = metadata["sample_name"].map(split_series)

    # Log the resulting split sizes
    train_size = sum(metadata["split"] == 0)
    val_size = sum(metadata["split"] == 1)
    test_size = sum(metadata["split"] == 2)
    logging.info("Data split: train %d, val %d, test %d", train_size, val_size, test_size)

    return metadata


# BAG RELATED FUNCTIONS
def parse_bag_size(value):
    """Parses bag size string into min and max values."""
    try:
        if "-" in str(value):
            min_val, max_val = map(int, str(value).split("-"))
            if min_val > max_val:
                raise ValueError("Invalid range: min > max")
            return [min_val, max_val]
        val = int(float(str(value)))
        return [val, val]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid bag_size format: {value}, error: {e}")

def parse_by_sample(value):
    """Parses by_sample string into a set of split values."""
    try:
        value = str(value)
        splits = [int(x) for x in value.split(",")]
        valid_splits = {0, 1, 2}
        if not all(x in valid_splits for x in splits):
            logging.warning("Invalid splits in by_sample: %s", splits)
            return None
        return splits
    except (ValueError, AttributeError):
        logging.warning("Failed to parse by_sample: %s", value)
        return None


def aggregate_embeddings(embeddings, pooling_method, use_gpu=False):
    # Convert embeddings to a float32 array explicitly.
    embeddings = np.asarray(embeddings, dtype=np.float32)
    
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    elif embeddings.ndim == 0:
        embeddings = embeddings.reshape(1, 1)
    
    logging.debug("Aggregating embeddings with shape: %s", embeddings.shape)
    
    if pooling_method == "max_pooling":
        result = np.max(embeddings, axis=0)
    elif pooling_method == "mean_pooling":
        result = np.mean(embeddings, axis=0)
    elif pooling_method == "sum_pooling":
        result = np.sum(embeddings, axis=0)
    elif pooling_method == "min_pooling":
        result = np.min(embeddings, axis=0)
    elif pooling_method == "median_pooling":
        result = np.median(embeddings, axis=0)
    elif pooling_method == "l2_norm_pooling":
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if norm.any():
            result = np.mean(embeddings / (norm + 1e-8), axis=0)
        else:
            result = np.mean(embeddings, axis=0)
    elif pooling_method == "geometric_mean_pooling":
        clipped = np.clip(embeddings, 1e-10, None)
        result = np.exp(np.mean(np.log(clipped), axis=0))
    elif pooling_method == "first_embedding":
        result = embeddings[0]
    elif pooling_method == "last_embedding":
        result = embeddings[-1]
    elif pooling_method == "attention_pooling":
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        with torch.no_grad():
            linear = nn.Linear(tensor.shape[1], 1).to(device)
            weights = nn.Softmax(dim=0)(linear(tensor))
            result = torch.sum(weights * tensor, dim=0).cpu().detach().numpy()
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")

    logging.debug("Aggregated embedding shape: %s", result.shape)
    return result


def bag_by_sample_process(group, pooling_method, bag_size, use_gpu):
    """Processes bags by sample."""
    embeddings = group.drop(columns=["sample_name", "label", "split"]).values
    logging.debug("Embeddings shape for sample %s: %s", group["sample_name"].iloc[0], embeddings.shape)
    sample_names = group["sample_name"].values
    labels = group["label"].values
    split = group["split"].iloc[0]
    num_instances = len(group)
    min_bag_size, max_bag_size = bag_size
    random_bag_size = np.random.randint(min_bag_size, max_bag_size + 1)
    num_bags = (num_instances + random_bag_size - 1) // random_bag_size

    logging.info("Sample %s: %d instances, %d bags",
                 group["sample_name"].iloc[0], num_instances, num_bags)

    bags = []
    for i in range(num_bags):
        start_idx = i * random_bag_size
        end_idx = min(start_idx + random_bag_size, num_instances)
        bag_embeddings = embeddings[start_idx:end_idx]
        bag_sample_names = sample_names[start_idx:end_idx]
        bag_labels = labels[start_idx:end_idx]
        aggregated_embeddings = aggregate_embeddings(bag_embeddings, pooling_method, use_gpu)
        logging.debug("Bag embedding shape in by_sample: %s", aggregated_embeddings.shape)
        bag_label = int(any(bag_labels == 1))
        bags.append({
            "bag_label": bag_label,
            "split": split,
            "bag_size": len(bag_sample_names),
            "bag_samples": list(bag_sample_names),
            "embedding": aggregated_embeddings,
        })
    logging.info("Created %d bags for sample: %s",
                 len(bags), group["sample_name"].iloc[0])
    return bags

def bag_turns_process(df_np, bag_sizes, pooling_method, use_gpu,
                      target_label=None, target_count=None):
    """Processes bags by turns."""
    indices_0 = np.where(df_np[:, 1] == 0)[0]
    indices_1 = np.where(df_np[:, 1] == 1)[0]
    np.random.shuffle(indices_0)
    np.random.shuffle(indices_1)
    bags = []
    bag_set = set()
    bag_count = 0
    min_bag_size, max_bag_size = bag_sizes

    while (len(indices_0) > 0 or len(indices_1) > 0) and \
          (target_count is None or bag_count < target_count):
        bag_size = np.random.randint(min_bag_size, max_bag_size + 1)
        if target_label == 0:
            num_0_samples = min(bag_size, len(indices_0))
            selected_indices = indices_0[:num_0_samples]
            indices_0 = indices_0[num_0_samples:]
        elif target_label == 1:
            num_1_samples = min(bag_size, len(indices_1))
            selected_indices = indices_1[:num_1_samples]
            indices_1 = indices_1[num_1_samples:]
        else:
            if len(indices_0) == 0:
                num_1_samples = min(bag_size, len(indices_1))
                selected_indices = indices_1[:num_1_samples]
                indices_1 = indices_1[num_1_samples:]
            elif len(indices_1) == 0:
                num_0_samples = min(bag_size, len(indices_0))
                selected_indices = indices_0[:num_0_samples]
                indices_0 = indices_0[num_0_samples:]
            else:
                num_1_samples = min(np.random.randint(1, bag_size + 1),
                                    len(indices_1))
                selected_indices_1 = indices_1[:num_1_samples]
                indices_1 = indices_1[num_1_samples:]
                num_0_samples = min(bag_size - num_1_samples, len(indices_0))
                selected_indices_0 = indices_0[:num_0_samples]
                indices_0 = indices_0[num_0_samples:]
                selected_indices = np.concatenate([selected_indices_0,
                                                   selected_indices_1])

        bag_data = df_np[selected_indices]
        if len(bag_data) > 0:
            only_embeddings = bag_data[:, 3:]
            aggregated_embedding = aggregate_embeddings(
                only_embeddings, pooling_method, use_gpu)
            logging.debug("Bag embedding shape in turns: %s", aggregated_embedding.shape)
            bag_label = int(any(bag_data[:, 1] == 1))
            bag_key = (tuple(map(tuple, only_embeddings)),
                       len(bag_data), tuple(bag_data[:, 0]))
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
    logging.info("Created %d bags for split: %d", len(bags), df_np[0, 2])
    return bags

def bag_random_process(df_np, bag_sizes, pooling_method, use_gpu,
                       target_label=None, target_count=None):
    """Processes bags randomly."""
    np.random.shuffle(df_np)
    min_bag_size, max_bag_size = bag_sizes
    bags = []
    bag_set = set()
    bag_count = 0
    idx = 0

    while idx < len(df_np) and (target_count is None or
                                bag_count < target_count):
        bag_size = np.random.randint(min_bag_size, max_bag_size + 1)
        end_idx = min(idx + bag_size, len(df_np))
        bag_data = df_np[idx:end_idx]
        only_embeddings = bag_data[:, 3:]
        aggregated_embedding = aggregate_embeddings(
            only_embeddings, pooling_method, use_gpu)
        logging.debug("Bag embedding shape in random: %s", aggregated_embedding.shape)
        bag_label = int(any(bag_data[:, 1] == 1))
        if target_label is not None and bag_label != target_label:
            idx = end_idx
            continue
        bag_key = (tuple(map(tuple, only_embeddings)),
                   len(bag_data), tuple(bag_data[:, 0]))
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

    logging.info("Created %d bags for split: %d", len(bags), df_np[0, 2])
    return bags


def write_csv(output_csv, list_embeddings, chunk_size=10000,
              append=False, ludwig_format=False):
    """Writes bags to a CSV file."""
    mode = "a" if append and os.path.exists(output_csv) else "w"
    try:
        with open(output_csv, mode=mode, encoding='utf-8', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
            headers = ["bag_samples", "bag_size", "bag_label", "split"]

            # Handle empty input
            if not list_embeddings:
                if mode == "w":
                    csv_writer.writerow(headers)
                logging.info("No bags generated, wrote empty headers")
                return

            first_bag = list_embeddings[0]
            if ludwig_format:
                if not isinstance(first_bag["embedding"], str):
                    raise ValueError("Embedding must be a string for Ludwig format")
                headers.append("embedding")
            else:
                if not isinstance(first_bag["embedding"], np.ndarray):
                    raise ValueError("Embedding must be NumPy array if not Ludwig")
                headers.extend([f"vector{i + 1}" for i in range(len(first_bag["embedding"]))])

            if mode == "w":
                csv_writer.writerow(headers)

            for i in range(0, len(list_embeddings), chunk_size):
                chunk = list_embeddings[i:i + chunk_size]
                rows = []
                for bag in chunk:
                    row = [",".join(map(str, bag["bag_samples"])),
                           bag["bag_size"], bag["bag_label"], bag["split"]]
                    if ludwig_format:
                        row.append(bag["embedding"])  # Use the string directly
                    else:
                        row.extend(bag["embedding"].tolist())  # Expand NumPy array
                    rows.append(row)
                csv_writer.writerows(rows)
                logging.info("Wrote %d bags to %s", len(rows), output_csv)
    except Exception as e:
        logging.error("Error writing to %s: %s", output_csv, e)
        raise


def generate_additional_bags(split_metadata, target_label, need, bag_sizes, pooling_method,
                             use_gpu, embeddings_path):
    """Generate additional bags for a specific label."""
    split_df = pd.merge(split_metadata[split_metadata["label"] == target_label],
                        pd.read_csv(embeddings_path), on="sample_name")
    if split_df.empty:
        logging.warning("No samples available to generate bags for label %d", target_label)
        return []
    df_np = split_df.to_numpy()
    return bag_turns_process(df_np, bag_sizes, pooling_method, use_gpu, target_label, need)


def balance_bags_split(bags, imbalance_cap, split_metadata, bag_sizes, pooling_method,
                       use_gpu, embeddings_path):
    """Balance bags for a split according to imbalance cap."""
    bags_0 = [b for b in bags if b["bag_label"] == 0]
    bags_1 = [b for b in bags if b["bag_label"] == 1]
    num_0, num_1 = len(bags_0), len(bags_1)
    imbalance = abs(num_0 - num_1) / max(num_0 + num_1, 1) * 100
    split = bags[0]["split"] if bags else split_metadata["split"].iloc[0]
    logging.info("Initial imbalance for split %d: %.2f%% (0: %d, 1: %d)",
                 split, imbalance, num_0, num_1)

    if imbalance <= imbalance_cap:
        return bags

    majority_count = max(num_0, num_1)
    minority_count = min(num_0, num_1)
    target_count = max(minority_count, int(majority_count / (1 + imbalance_cap / 100)))
    target_count = max(target_count, 1)  # Ensure at least one bag

    if num_0 < num_1:  # Label 0 is minority
        need = target_count - num_0
        logging.info("Target count: %d, need %d bags for label 0", target_count, need)
        extra_bags = generate_additional_bags(split_metadata, 0, need, bag_sizes,
                                              pooling_method, use_gpu, embeddings_path)
        bags.extend(extra_bags[:need])
    elif num_1 < num_0:  # Label 1 is minority
        need = target_count - num_1
        logging.info("Target count: %d, need %d bags for label 1", target_count, need)
        extra_bags = generate_additional_bags(split_metadata, 1, need, bag_sizes,
                                              pooling_method, use_gpu, embeddings_path)
        bags.extend(extra_bags[:need])

    final_0, final_1 = len([b for b in bags if b["bag_label"] == 0]), len([b for b in bags if b["bag_label"] == 1])
    final_imbalance = abs(final_0 - final_1) / max(final_0 + final_1, 1) * 100
    logging.info("Final imbalance for split %d: %.2f%% (0: %d, 1: %d)",
                 split, final_imbalance, final_0, final_1)
    return bags


def truncate_bags(split_bags):
    """Truncates bags to balance the number of positive and negative bags, retaining at least one bag per label if possible."""
    # Separate bags by label
    bags_0 = [bag for bag in split_bags if bag["bag_label"] == 0]
    bags_1 = [bag for bag in split_bags if bag["bag_label"] == 1]

    # Case 1: No bags for either label
    if len(bags_0) == 0 and len(bags_1) == 0:
        logging.warning("No bags for either label in split. Returning empty list.")
        return []

    # Case 2: No bags with label 0, keep all label 1 bags
    elif len(bags_0) == 0:
        logging.info("No bags with label 0. Retaining all bags with label 1.")
        return bags_1

    # Case 3: No bags with label 1, keep all label 0 bags
    elif len(bags_1) == 0:
        logging.info("No bags with label 1. Retaining all bags with label 0.")
        return bags_0

    # Case 4: Both labels have bags, truncate to minority count
    else:
        minority_count = min(len(bags_0), len(bags_1))
        truncated_bags = bags_0[:minority_count] + bags_1[:minority_count]
        logging.info("Truncated to %d bags per label.", minority_count)
        return truncated_bags


def transform_bags_for_ludwig(bags):
    """Transforms bags into Ludwig format."""
    return [dict(bag, embedding=" ".join(map(str, bag["embedding"])))
            for bag in bags]


def process_split(split, metadata, pooling_method, balance_enforced,
                  bag_sizes, use_gpu, split_temp_files, by_sample_set,
                  embeddings_path, imbalance_cap, do_truncate_bags,
                  split_bag_output):
    """Processes a single split and returns bags."""
    logging.info("Processing split: %d", split)
    split_metadata = metadata[metadata["split"] == split]
    split_temp_file = split_temp_files.get(split)
    if not split_temp_file or not os.path.exists(split_temp_file):
        logging.warning("No temp file for split %d", split)
        return []
    split_embeddings = pd.read_csv(split_temp_file)
    split_bags = []

    # Bag creation logic (unchanged)
    if split in by_sample_set:
        for sample_name in split_embeddings["sample_name"].unique():
            sample_chunk = split_embeddings[
                split_embeddings["sample_name"] == sample_name]
            sample_df = pd.merge(split_metadata[
                split_metadata["sample_name"] == sample_name],
                sample_chunk, on="sample_name")
            if not sample_df.empty:
                bags = bag_by_sample_process(sample_df, pooling_method,
                                             bag_sizes, use_gpu)
                split_bags.extend(bags)
    else:
        split_df = pd.merge(split_metadata, split_embeddings,
                            on="sample_name")
        if not split_df.empty:
            df_np = split_df.to_numpy()
            if balance_enforced:
                bags = bag_turns_process(df_np, bag_sizes, pooling_method,
                                         use_gpu)
            else:
                bags = bag_random_process(df_np, bag_sizes, pooling_method,
                                          use_gpu)
            split_bags.extend(bags)

    # Apply imbalance cap if specified
    if split_bags and imbalance_cap is not None:
        split_bags = balance_bags_split(
            split_bags, imbalance_cap, split_metadata, bag_sizes,
            pooling_method, use_gpu, embeddings_path)  # Fixed: 7 arguments only

    # Apply truncation if enabled
    if do_truncate_bags:
        split_bags = truncate_bags(split_bags)

    # Write bags to CSV only if there are bags
    if split_bags:  # New check
        try:
            with open(split_bag_output, "a", newline="") as sf:
                writer = csv.DictWriter(sf, fieldnames=split_bags[0].keys())
                if os.stat(split_bag_output).st_size == 0:
                    writer.writeheader()
                writer.writerows(split_bags)
        except Exception as e:
            logging.error("Error writing to %s: %s", split_bag_output, e)
            raise
    else:
        logging.info("No bags to write for split %d after processing", split)

    logging.info("Finished split %d with %d bags", split, len(split_bags))
    return split_bags


def validate_metadata(metadata):
    """Validates metadata for required columns."""
    required_cols = {"sample_name", "label"}
    if not required_cols.issubset(metadata.columns):
        missing = required_cols - set(metadata.columns)
        raise ValueError(f"Metadata missing columns: {missing}")
    return metadata

def setup_temp_files(splits):
    """Sets up temporary files for splits and bag outputs."""
    split_temp_files = {
        split: tempfile.mkstemp(prefix=f"split_{split}_",
                                suffix=".csv", dir=os.getcwd())[1]
        for split in splits
    }
    split_bag_outputs = {
        split: tempfile.mkstemp(prefix=f"split_bags_{split}_",
                                suffix=".csv", dir=os.getcwd())[1]
        for split in splits
    }
    return split_temp_files, split_bag_outputs

def distribute_embeddings(embeddings_path, metadata, split_temp_files):
    """Distributes embeddings to temporary split files."""
    buffer_size = 50000
    buffers = {temp_file: [] for temp_file in split_temp_files.values()}
    logging.info("Distributing embeddings from %s to temp files",
                 embeddings_path)
    sample_to_split = dict(zip(metadata["sample_name"], metadata["split"]))

    try:
        for chunk in pd.read_csv(embeddings_path, chunksize=50000):
            for _, row in chunk.iterrows():
                sample_name = row["sample_name"]
                if sample_name in sample_to_split:
                    split = sample_to_split[sample_name]
                    temp_file = split_temp_files.get(split)
                    if temp_file:
                        buffers[temp_file].append(row.tolist())
                        if len(buffers[temp_file]) >= buffer_size:
                            with open(temp_file, "a", newline="") as tf:
                                writer = csv.writer(tf)
                                if os.stat(temp_file).st_size == 0:
                                    writer.writerow(chunk.columns.tolist())
                                writer.writerows(buffers[temp_file])
                            buffers[temp_file] = []
    except Exception as e:
        logging.error("Error distributing embeddings: %s", e)
        raise

    for temp_file, rows in buffers.items():
        if rows:
            try:
                with open(temp_file, "a", newline="") as tf:
                    writer = csv.writer(tf)
                    if os.stat(temp_file).st_size == 0:
                        writer.writerow(chunk.columns.tolist())
                    writer.writerows(rows)
            except Exception as e:
                logging.error("Error writing buffer to %s: %s", temp_file, e)
                raise


def process_splits(splits, metadata, pooling_method, balance_enforced,
                   bag_sizes, use_gpu, split_temp_files, by_sample_set,
                   embeddings_path, imbalance_cap, do_truncate_bags,
                   split_bag_outputs):
    """Processes splits in parallel and returns all bags."""
    with mp.Pool(processes=mp.cpu_count()) as pool:
        args = [(split, metadata, pooling_method, balance_enforced,
                 bag_sizes, use_gpu, split_temp_files, by_sample_set,
                 embeddings_path, imbalance_cap, do_truncate_bags,
                 split_bag_outputs[split]) for split in splits]
        logging.info("Arguments to process_split: %s", args[0][1])  # Log first args for debugging
        results = pool.starmap(process_split, args)
    all_bags = [bag for result in results for bag in result]
    logging.info("Collected %d total bags from all splits", len(all_bags))
    return all_bags


def write_final_output(output_csv, all_bags, ludwig_format):
    """Writes the final output to CSV."""
    if all_bags:
        if ludwig_format:
            all_bags = transform_bags_for_ludwig(all_bags)
        write_csv(output_csv, all_bags, append=False,
                  ludwig_format=ludwig_format)
        logging.info("Completed bagging, wrote to %s", output_csv)
    else:
        logging.info("No bags to write to CSV")

def cleanup_temp_files(split_temp_files, split_bag_outputs):
    """Cleans up temporary files."""
    for temp_file in split_temp_files.values():
        try:
            os.remove(temp_file)
            logging.info("Cleaned up temp file: %s", temp_file)
        except Exception as e:
            logging.error("Error removing %s: %s", temp_file, e)
    for split_bag_output in split_bag_outputs.values():
        try:
            os.remove(split_bag_output)
            logging.info("Cleaned up temp bag file: %s", split_bag_output)
        except Exception as e:
            logging.error("Error removing %s: %s", split_bag_output, e)

def bag_processing(embeddings_path, metadata, pooling_method,
                   balance_enforced, bag_sizes, repeats, ludwig_format,
                   by_sample, use_gpu, output_csv, imbalance_cap,
                   do_truncate_bags):
    """Processes bags from embeddings and metadata."""
    bag_sizes = parse_bag_size(bag_sizes)
    if not isinstance(bag_sizes, list) or len(bag_sizes) != 2:
        raise ValueError(f"bag_sizes must be a list of two integers, got: {bag_sizes}")
    metadata = validate_metadata(metadata)
    splits = metadata["split"].unique()
    logging.info("Processing %d splits: %s", len(splits), list(splits))
    by_sample_set = set(by_sample) if by_sample else set()
    split_temp_files, split_bag_outputs = setup_temp_files(splits)
    all_bags = []

    imbalance_cap = None if imbalance_cap in (None, "", "None") else int(imbalance_cap)

    try:
        for _ in range(max(1, repeats)):
            distribute_embeddings(embeddings_path, metadata, split_temp_files)
            bags = process_splits(splits, metadata, pooling_method,
                                  balance_enforced, bag_sizes, use_gpu,
                                  split_temp_files, by_sample_set,
                                  embeddings_path, imbalance_cap,
                                  do_truncate_bags, split_bag_outputs)
            all_bags.extend(bags)
        write_final_output(output_csv, all_bags, ludwig_format)
    finally:
        cleanup_temp_files(split_temp_files, split_bag_outputs)


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' to support CUDA
    mp.set_start_method('spawn', force=True)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(
        description="Create bags from embeddings and metadata")
    parser.add_argument("--embeddings_csv", type=str, required=True,
                        help="Path to embeddings CSV")
    parser.add_argument("--metadata_csv", type=str, required=True,
                        help="Path to metadata CSV")
    parser.add_argument("--split_proportions", type=str,
                        default='0.7,0.1,0.2',
                        help="Proportions for train, val, test splits")
    parser.add_argument("--dataleak", action="store_true",
                        help="Prevents data leakage")
    parser.add_argument("--balance_enforced", action="store_true",
                        help="Enforce balanced bagging")
    parser.add_argument("--bag_size", type=str, required=True,
                        help="Bag size (e.g., '4' or '3-5')")
    parser.add_argument("--pooling_method", type=str, required=True,
                        help="Pooling method")
    parser.add_argument("--by_sample", type=parse_by_sample, default=None,
                        help="Splits to bag by sample")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of bagging repeats")
    parser.add_argument("--ludwig_format", action="store_true",
                        help="Output in Ludwig format")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to output CSV")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--imbalance_cap", type=int, default=None,
                        help="Max imbalance percentage")
    parser.add_argument("--truncate_bags", action="store_true",
                        help="Truncate bags for balance")
    parser.add_argument("--use_gpu", action="store_true", default=False,
                        help="Use GPU for pooling")
    args = parser.parse_args()
    print(f"Starting bagging with args: {args}", flush=True)

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
        metadata_csv = split_data(
            metadata_csv, split_proportions=args.split_proportions,
            dataleak=args.dataleak)
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
        do_truncate_bags=args.truncate_bags)
