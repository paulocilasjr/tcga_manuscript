import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import argparse
import seaborn as sns
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot K-fold training metrics from a JSON file.')
parser.add_argument('json_file', type=str, help='Path to the JSON file containing training statistics')
parser.add_argument('--output_file', type=str, default='kfold_metrics_plot.png', 
                    help='Path to save the output figure (default: kfold_metrics_plot.png)')
args = parser.parse_args()

# Load the JSON file
with open(args.json_file, 'r') as f:
    data = json.load(f)

# Identify fold keys (e.g., 'fold_1', 'fold_2', ..., 'fold_n')
fold_keys = sorted([key for key in data.keys() if 'fold' in key])
num_folds = len(fold_keys)

# Define the metrics to extract
metrics = ["loss", "accuracy", "precision", "recall", "roc_auc", "specificity", "f1_score"]

# Compute the t-value for the 95% confidence interval (two-tailed, df = num_folds - 1)
t_value = t.ppf(0.975, df=num_folds - 1) if num_folds > 1 else 0

# Prepare data for plotting
plot_data = []
for metric in metrics:
    split_values = {"Train": [], "Evaluation": [], "Test": []}
    for fold_key in fold_keys:
        train_label_data = data[fold_key]['training']['label']
        test_label_data = data[fold_key]['test']['label']
        eval_label_data = data[fold_key]['fold_eval_stats']['label']
        
        if metric == "f1_score":
            train_precision, train_recall = train_label_data["precision"][-1], train_label_data["recall"][-1]
            test_precision, test_recall = test_label_data["precision"][-1], test_label_data["recall"][-1]
            eval_precision, eval_recall = eval_label_data["precision"], eval_label_data["recall"]
            
            train_value = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
            test_value = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
            eval_value = 2 * (eval_precision * eval_recall) / (eval_precision + eval_recall) if (eval_precision + eval_recall) > 0 else 0
        else:
            train_value = train_label_data[metric][-1]
            eval_value = eval_label_data[metric]
            test_value = test_label_data[metric][-1]
        
        split_values["Train"].append(train_value)
        split_values["Evaluation"].append(eval_value)
        split_values["Test"].append(test_value)
    
    # Compute mean and confidence interval for each split
    for split_name, values in split_values.items():
        mean_val = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(num_folds) if num_folds > 1 else 0
        ci = t_value * std_err if num_folds > 1 else 0
        plot_data.append({"Split": split_name, "Metric": metric, "Mean": mean_val, "CI": ci})

# Convert to DataFrame and filter out "Evaluation"
df = pd.DataFrame(plot_data)
df = df[df["Split"] != "Evaluation"]  # Exclude Evaluation split

# Create the grouped bar plot
plt.figure(figsize=(12, 7))
sns.barplot(
    x="Metric",
    y="Mean",
    hue="Split",
    data=df,
    palette="tab10",
    capsize=0.1,
    errorbar=None  # We'll add custom error bars manually
)

# Add custom error bars
for i, split in enumerate(df["Split"].unique()):
    split_data = df[df["Split"] == split]
    plt.errorbar(
        x=np.arange(len(split_data)) + i * 0.2 - 0.1,  # Adjusted offset for 2 splits
        y=split_data["Mean"],
        yerr=split_data["CI"],
        fmt='none',
        color='black',
        capsize=3,
        linewidth=1.5
    )

plt.ylabel("Value (0-1)")
plt.xlabel("Metrics")
plt.ylim(0, 1)
plt.title("K-Fold Training Metrics by Split with Confidence Intervals (Train and Test Only)")
plt.legend(title="Data Split")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a high-resolution figure using the provided argument
plt.savefig(args.output_file, dpi=300, bbox_inches='tight')

# Display the plot (optional, remove if you only want to save)
plt.show()
