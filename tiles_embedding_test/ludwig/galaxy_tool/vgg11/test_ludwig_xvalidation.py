import pandas as pd
import numpy as np
from ludwig.api import LudwigModel
from sklearn.model_selection import KFold
import yaml
import sys
import os
from scipy import stats

# Load the Ludwig config
with open("ludwig_config.yml", "r") as f:
    config = yaml.safe_load(f)

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate the mean and 95% confidence interval for a list of values."""
    mean = np.mean(data)
    if len(data) > 1:
        ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=stats.sem(data))
        return mean, ci[0], ci[1]
    return mean, mean, mean  # If only one fold, no CI

def run_cross_validation(data_path: str, k: int = 10):
    """Run k-fold cross-validation with Ludwig and report detailed metrics."""
    # Load dataset
    if not os.path.isfile(data_path):
        print(f"Error: File '{data_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    print(f"Loading dataset from: {data_path}", flush=True)
    df = pd.read_csv(data_path)
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns", flush=True)

    # Define k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    print(f"Performing {k}-fold cross-validation...", flush=True)

    # Store results for each fold
    results = {
        'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'roc_auc': [], 'specificity': [], 'f1': []},
        'validation': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'roc_auc': [], 'specificity': [], 'f1': []},
        'test': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'roc_auc': [], 'specificity': [], 'f1': []}
    }

    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        print(f"\n=== Fold {fold}/{k} ===", flush=True)
        
        # Split data into train and test sets
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Further split train into train and validation (80% train, 20% validation)
        val_split = int(0.8 * len(train_df))
        train_split_df = train_df.iloc[:val_split]
        val_split_df = train_df.iloc[val_split:]
        
        print(f"Train split: {len(train_split_df)} rows, Validation split: {len(val_split_df)} rows, Test split: {len(test_df)} rows", flush=True)
        
        # Create and train a Ludwig model
        model = LudwigModel(config)
        train_stats, _, _ = model.train(
            training_set=train_split_df,  # Fixed parameter name
            validation_set=val_split_df,  # Fixed parameter name
            output_directory=f"results_fold_{fold}"
        )
        
        # Evaluate on test set
        test_stats = model.evaluate(test_df, collect_predictions=True, collect_overall_stats=True)
        
        # Extract metrics for train, validation, and test
        label_stats_train = train_stats['training']['label']
        label_stats_val = train_stats['validation']['label']
        label_stats_test = test_stats[0]['label']

        # Metrics to extract
        metrics = ['loss', 'accuracy', 'precision', 'recall', 'roc_auc', 'specificity', 'f1']
        
        # Print and store train metrics
        print("\nTrain Metrics:", flush=True)
        for metric in metrics:
            value = label_stats_train.get(metric, label_stats_train.get(f'{metric}_score', 0.0))  # Handle F1 naming
            results['train'][metric].append(value)
            print(f"  {metric.capitalize()}: {value:.4f}", flush=True)
        
        # Print and store validation metrics
        print("\nValidation Metrics:", flush=True)
        for metric in metrics:
            value = label_stats_val.get(metric, label_stats_val.get(f'{metric}_score', 0.0))
            results['validation'][metric].append(value)
            print(f"  {metric.capitalize()}: {value:.4f}", flush=True)
        
        # Print and store test metrics
        print("\nTest Metrics:", flush=True)
        for metric in metrics:
            value = label_stats_test.get(metric, label_stats_test.get(f'{metric}_score', 0.0))
            results['test'][metric].append(value)
            print(f"  {metric.capitalize()}: {value:.4f}", flush=True)

    # Summarize results across folds
    print("\n=== Cross-Validation Summary ===", flush=True)
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.capitalize()} Metrics (Average and 95% CI):", flush=True)
        for metric in metrics:
            mean, ci_lower, ci_upper = calculate_confidence_interval(results[split][metric])
            print(f"  {metric.capitalize()}: {mean:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})", flush=True)

def main():
    """Main function to run cross-validation with a provided CSV file."""
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the CSV file as an argument.", file=sys.stderr)
        print("Usage: python test_ludwig_xvalidation.py <csv_file>", file=sys.stderr)
        sys.exit(1)

    data_path = sys.argv[1]
    run_cross_validation(data_path, k=10)

if __name__ == "__main__":
    main()
