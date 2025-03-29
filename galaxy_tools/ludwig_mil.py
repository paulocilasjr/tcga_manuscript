import argparse
import csv
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

# Configure logging
logging.basicConfig(
    filename="/tmp/ludwig_embeddings.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)

# [Previous functions unchanged: parse_bag_size, parse_by_sample, load_csv, 
# str_array_split, split_sizes, split_data, attention_pooling, gated_pooling, 
# aggregate_embeddings, bag_by_sample, bag_turns, bag_random remain the same]

def write_csv(output_csv, list_embeddings, chunk_size=10000, append=True):
    """Writes bags to a CSV file in chunks, with option to append."""
    if not list_embeddings:
        mode = "a" if append and os.path.exists(output_csv) else "w"
        with open(output_csv, mode=mode, encoding='utf-8', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if not append or not os.path.exists(output_csv):
                csv_writer.writerow(["bag_samples", "bag_size", "bag_label", "split"])
            logging.info("No valid data found for this chunk.")
        return

    first_item = list_embeddings[0]
    mode = "a" if append and os.path.exists(output_csv) else "w"
    with open(output_csv, mode=mode, encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
        headers = ["bag_samples", "bag_size", "bag_label", "split"]

        if isinstance(first_item["embedding"], str):
            headers.append("embedding")
        elif isinstance(first_item["embedding"], np.ndarray):
            embedding_size = len(first_item["embedding"])
            headers.extend([f"vector{i+1}" for i in range(embedding_size)])
        else:
            raise ValueError("Expected string or NumPy array.")

        if not append or not os.path.exists(output_csv):
            csv_writer.writerow(headers)

        for i in range(0, len(list_embeddings), chunk_size):
            chunk = list_embeddings[i:i + chunk_size]
            for bag in chunk:
                row = [",".join(map(str, bag["bag_samples"])),
                       bag["bag_size"],
                       bag["bag_label"],
                       bag["split"]]
                if isinstance(bag["embedding"], str):
                    row.append(bag["embedding"])
                else:
                    row.extend(bag["embedding"].tolist())
                csv_writer.writerow(row)

def balance_bags(split_bags, imbalance_cap, split_df, bag_sizes, pooling_method, balance_enforced, use_gpu, output_csv):
    """Balances the number of bags within the specified imbalance cap for a single split."""
    bags_0 = [bag for bag in split_bags if bag["bag_label"] == 0]
    bags_1 = [bag for bag in split_bags if bag["bag_label"] == 1]
    num_bags_0 = len(bags_0)
    num_bags_1 = len(bags_1)
    total_bags = num_bags_0 + num_bags_1

    if total_bags == 0:
        logging.info("No bags to balance in this split.")
        return split_bags

    imbalance = abs(num_bags_0 - num_bags_1) / min(num_bags_0, num_bags_1) * 100 if min(num_bags_0, num_bags_1) > 0 else float('inf')
    
    if imbalance <= imbalance_cap:
        logging.info(f"Split {split_bags[0]['split']}: Imbalance ({imbalance:.2f}%) is within cap ({imbalance_cap}%). No adjustment needed.")
        return split_bags

    target_diff = int(min(num_bags_0, num_bags_1) * imbalance_cap / 100)
    if num_bags_0 > num_bags_1:
        target_count = num_bags_0 - num_bags_1 - target_diff
        target_label = 1
    else:
        target_count = num_bags_1 - num_bags_0 - target_diff
        target_label = 0

    logging.info(f"Split {split_bags[0]['split']}: Adding {target_count} bags with label {target_label} to reduce imbalance.")

    if balance_enforced:
        extra_bags = bag_turns(split_df, bag_sizes, pooling_method, repeats=1, use_gpu=use_gpu, target_label=target_label, target_count=target_count)
    else:
        extra_bags = bag_random(split_df, bag_sizes, pooling_method, repeats=1, use_gpu=use_gpu, target_label=target_label, target_count=target_count)

    if extra_bags:
        split_bags.extend(extra_bags)
        logging.info(f"Split {split_bags[0]['split']}: Added {len(extra_bags)} extra bags with label {target_label}.")
    
    return split_bags

def truncate_bags(split_bags):
    """Truncates the bags of the majority label to match the minority label count for a single split."""
    bags_0 = [bag for bag in split_bags if bag["bag_label"] == 0]
    bags_1 = [bag for bag in split_bags if bag["bag_label"] == 1]
    num_bags_0 = len(bags_0)
    num_bags_1 = len(bags_1)

    if num_bags_0 == num_bags_1:
        logging.info(f"Split {split_bags[0]['split']}: No truncation needed; bag counts are equal ({num_bags_0} each).")
        return split_bags

    minority_count = min(num_bags_0, num_bags_1)
    if num_bags_0 > num_bags_1:
        bags_0 = bags_0[:minority_count]
        logging.info(f"Split {split_bags[0]['split']}: Truncated {num_bags_0 - minority_count} bags with label 0 to match {minority_count} bags with label 1.")
    else:
        bags_1 = bags_1[:minority_count]
        logging.info(f"Split {split_bags[0]['split']}: Truncated {num_bags_1 - minority_count} bags with label 1 to match {minority_count} bags with label 0.")

    return bags_0 + bags_1

def bag_processing(embeddings_path,
                   metadata,
                   pooling_method,
                   balance_enforced=False,
                   bag_sizes=[3, 5],
                   repeats=1,
                   ludwig_format=False,
                   by_sample=None,
                   use_gpu=False,
                   output_csv=None,
                   imbalance_cap=None,
                   truncate_bags=False):
    """Processes embeddings and metadata to create bags, handling each split independently."""
    bag_sizes = parse_bag_size(bag_sizes)

    required_cols = {"sample_name", "label"}
    if not required_cols.issubset(metadata.columns):
        missing = required_cols - set(metadata.columns)
        raise ValueError(f"Metadata CSV missing required columns: {missing}")

    split_dfs = {}
    for split in metadata['split'].unique():
        split_metadata = metadata[metadata['split'] == split]
        split_sample_names = split_metadata['sample_name'].unique()
        split_bags = []

        if by_sample is not None and split in by_sample:
            for sample_name in split_sample_names:
                sample_metadata = split_metadata[split_metadata['sample_name'] == sample_name]
                sample_chunks = []
                for chunk in pd.read_csv(embeddings_path, chunksize=500000):
                    chunk_filtered = chunk[chunk['sample_name'] == sample_name]
                    if not chunk_filtered.empty:
                        sample_chunks.append(chunk_filtered)
                del chunk

                if sample_chunks:
                    sample_embeddings = pd.concat(sample_chunks)
                    logging.info(f"Loaded {len(sample_embeddings)} embeddings for sample_name: {sample_name} in split {split}")
                    sample_df = pd.merge(sample_metadata, sample_embeddings, on='sample_name')
                    if sample_df.empty:
                        logging.warning(f"No matching embeddings found for sample_name: {sample_name} in split {split}")
                        continue
                    bags = bag_by_sample(sample_df, pooling_method, bag_sizes, use_gpu)
                    split_bags.extend(bags)
                else:
                    logging.warning(f"No embeddings found for sample_name: {sample_name} in split {split}")
        else:
            split_embeddings_chunks = []
            for chunk in pd.read_csv(embeddings_path, chunksize=500000):
                chunk_filtered = chunk[chunk['sample_name'].isin(split_sample_names)]
                if not chunk_filtered.empty:
                    split_embeddings_chunks.append(chunk_filtered)
            del chunk

            if split_embeddings_chunks:
                split_embeddings = pd.concat(split_embeddings_chunks)
                logging.info(f"Loaded {len(split_embeddings)} embeddings for split {split}")
                split_df = pd.merge(split_metadata, split_embeddings, on='sample_name')
                if split_df.empty:
                    logging.warning(f"No matching embeddings found for split {split}")
                    continue
                split_dfs[split] = split_df
                if balance_enforced:
                    bags = bag_turns(split_df, bag_sizes, pooling_method, repeats, use_gpu)
                else:
                    bags = bag_random(split_df, bag_sizes, pooling_method, repeats, use_gpu)
                split_bags.extend(bags)
            else:
                logging.warning(f"No embeddings found for split {split}")

        if split_bags:
            # Apply imbalance_cap if specified
            if imbalance_cap is not None and split in split_dfs:
                split_bags = balance_bags(split_bags, imbalance_cap, split_dfs[split], bag_sizes, pooling_method, balance_enforced, use_gpu, output_csv)

            # Apply truncate_bags if specified
            if truncate_bags:
                split_bags = truncate_bags(split_bags)

            # Transform and write bags for this split
            if ludwig_format:
                split_bags = transform_bags_for_ludwig(split_bags)
            write_csv(output_csv, split_bags, append=True)
            del split_bags  # Clear memory after writing

    # Return an empty list since all bags are written to CSV
    return []

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
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--imbalance_cap", type=int, default=None, help="Max percentage imbalance between positive and negative bags")
    parser.add_argument("--truncate_bags", action="store_true", help="Truncate excess bags to match minority label count")

    args = parser.parse_args()

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    metadata_csv = load_csv(args.metadata_csv)
    if "split" not in metadata_csv.columns:
        metadata_csv = split_data(metadata_csv, split_proportions=args.split_proportions, dataleak=args.dataleak)

    processed_embeddings = bag_processing(
        args.embeddings_csv,
        metadata_csv,
        args.pooling_method,
        args.balance_enforced,
        args.bag_size,
        args.repeats,
        args.ludwig_format,
        args.by_sample,
        output_csv=args.output_csv,
        imbalance_cap=args.imbalance_cap,
        truncate_bags=args.truncate_bags
    )
