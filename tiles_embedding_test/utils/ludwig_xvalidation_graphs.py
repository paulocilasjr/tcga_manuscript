import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# Load the JSON file
with open('kfold_training_statistics.json', 'r') as f:
    data = json.load(f)

# Identify fold keys (e.g., 'fold_1', 'fold_2', ..., 'fold_n')
fold_keys = sorted([key for key in data.keys() if 'fold' in key])
num_folds = len(fold_keys)

# Define the metrics to extract
metrics = ["loss", "accuracy", "precision", "recall", "roc_auc", "specificity"]

# Compute the t-value for the 95% confidence interval (two-tailed, df = num_folds - 1)
t_value = t.ppf(0.975, df=num_folds - 1)

# Create a figure with one subplot per metric (including F1-Score)
fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(5 * (len(metrics) + 1), 5))
if len(metrics) == 1:
    axes = [axes]

# Process each metric and create a subplot
for i, metric in enumerate(metrics + ["f1_score"]):
    ax = axes[i]
    train_values, eval_values, test_values = [], [], []
    
    for fold_key in fold_keys:
        train_label_data = data[fold_key]['training']['label']
        test_label_data = data[fold_key]['test']['label']
        eval_label_data = data[fold_key]['fold_eval_stats']['label']
        
        if metric == "f1_score":
            # Calculate F1-Score from Precision and Recall
            train_precision = train_label_data["precision"][-1]
            train_recall = train_label_data["recall"][-1]
            test_precision = test_label_data["precision"][-1]
            test_recall = test_label_data["recall"][-1]
            eval_precision = eval_label_data["precision"]
            eval_recall = eval_label_data["recall"]
            
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall)
            test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
            eval_f1 = 2 * (eval_precision * eval_recall) / (eval_precision + eval_recall)
            
            train_values.append(train_f1)
            eval_values.append(eval_f1)
            test_values.append(test_f1)
        else:
            train_values.append(train_label_data[metric][-1])  # Last value of training metric
            eval_values.append(eval_label_data[metric])  # Single eval value
            test_values.append(test_label_data[metric][-1])  # Last value of test metric
    
    # Calculate means and standard deviations
    train_mean, eval_mean, test_mean = np.mean(train_values), np.mean(eval_values), np.mean(test_values)
    train_std, eval_std, test_std = np.std(train_values, ddof=1), np.std(eval_values, ddof=1), np.std(test_values, ddof=1)
    
    # Compute the margin of error for the 95% confidence interval
    train_err, eval_err, test_err = (t_value * train_std / np.sqrt(num_folds),
                                     t_value * eval_std / np.sqrt(num_folds),
                                     t_value * test_std / np.sqrt(num_folds))
    
    # Prepare data for plotting
    labels, means, errors = ['Train', 'Evaluation', 'Test'], [train_mean, eval_mean, test_mean], [train_err, eval_err, test_err]
    
    # Plot bars with error bars
    ax.bar(labels, means, yerr=errors, capsize=5)
    ax.set_title(metric.replace("_", " ").capitalize())
    ax.set_ylabel('Value')
    
    # Add mean values above each bar
    for j, mean in enumerate(means):
        ax.text(j, mean + 0.01 * max(means), f'{mean:.3f}', ha='center')

# Adjust layout to prevent overlap and display the plot
plt.tight_layout()
plt.show()

