import matplotlib.pyplot as plt
import numpy as np
import json

# Load data from JSON file
with open('metrics_to_plot.json', 'r') as f:
    data = json.load(f)

# Extract data from JSON
results = [entry['model'] for entry in data]
loss = [entry['loss']['value'] for entry in data]
loss_ci_lower = [entry['loss']['ci_lower'] for entry in data]
loss_ci_upper = [entry['loss']['ci_upper'] for entry in data]

accuracy = [entry['accuracy']['value'] for entry in data]
accuracy_ci_lower = [entry['accuracy']['ci_lower'] for entry in data]
accuracy_ci_upper = [entry['accuracy']['ci_upper'] for entry in data]

precision = [entry['precision']['value'] for entry in data]
precision_ci_lower = [entry['precision']['ci_lower'] for entry in data]
precision_ci_upper = [entry['precision']['ci_upper'] for entry in data]

recall = [entry['recall']['value'] for entry in data]
recall_ci_lower = [entry['recall']['ci_lower'] for entry in data]
recall_ci_upper = [entry['recall']['ci_upper'] for entry in data]

roc_auc = [entry['roc_auc']['value'] for entry in data]
roc_auc_ci_lower = [entry['roc_auc']['ci_lower'] for entry in data]
roc_auc_ci_upper = [entry['roc_auc']['ci_upper'] for entry in data]

specificity = [entry['specificity']['value'] for entry in data]
specificity_ci_lower = [entry['specificity']['ci_lower'] for entry in data]
specificity_ci_upper = [entry['specificity']['ci_upper'] for entry in data]

f1_score = [entry['f1_score']['value'] for entry in data]
f1_score_ci_lower = [entry['f1_score']['ci_lower'] for entry in data]
f1_score_ci_upper = [entry['f1_score']['ci_upper'] for entry in data]

# Define metrics and associated data
metrics = [accuracy, precision, recall, roc_auc, specificity, f1_score]
metric_names = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'Specificity', 'F1-Score']
ci_lower = [accuracy_ci_lower, precision_ci_lower, recall_ci_lower, roc_auc_ci_lower, specificity_ci_lower, f1_score_ci_lower]
ci_upper = [accuracy_ci_upper, precision_ci_upper, recall_ci_upper, roc_auc_ci_upper, specificity_ci_upper, f1_score_ci_upper]

# Colors for bars and label pairs
colors = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F']
loss_color = '#B3B3B3'
label_colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728']  # Colors for x-axis label pairs

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

bar_width = 0.12
x = np.arange(len(results))
n_metrics = len(metrics)

# Plot bars for each metric
for i, (metric, name, color, lower, upper) in enumerate(zip(metrics, metric_names, colors, ci_lower, ci_upper)):
    offset = (i - n_metrics / 2) * bar_width
    bars = ax1.bar(x + offset, metric, bar_width, label=name, color=color, edgecolor='white', linewidth=0.5)
    yerr_lower = [m - l if l is not None else 0 for m, l in zip(metric, lower)]
    yerr_upper = [u - m if u is not None else 0 for m, u in zip(metric, upper)]
    yerr = np.array([yerr_lower, yerr_upper])
    ax1.errorbar(x + offset, metric, yerr=yerr, fmt='none', color='black', capsize=3, linewidth=1)

# Plot loss bars
loss_offset = (n_metrics / 2) * bar_width + bar_width * 0.5
bars = ax2.bar(x + loss_offset, loss, bar_width, label='Loss', color=loss_color, edgecolor='white', linewidth=0.5)
yerr_lower = [l - ll if ll is not None else 0 for l, ll in zip(loss, loss_ci_lower)]
yerr_upper = [lu - l if lu is not None else 0 for l, lu in zip(loss, loss_ci_upper)]
yerr = np.array([yerr_lower, yerr_upper])
ax2.errorbar(x + loss_offset, loss, yerr=yerr, fmt='none', color='black', capsize=3, linewidth=1)

# Add threshold line
ax1.axhline(y=0.7, color='red', linestyle=':', linewidth=1, label='Threshold 0.7')

# Customize axes
ax1.set_xlabel('Models', fontsize=12)
ax1.set_ylabel('Performance Metrics Evaluation', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax1.set_xticks(x)
ax1.set_ylim(0, 1.1)
ax2.set_ylim(0, 3)

# Set x-axis labels with line breaks and pair-wise coloring
xticklabels = [label.replace(' ', '\n') for label in results]
ax1.set_xticklabels([])  # Clear default labels
for i, label in enumerate(xticklabels):
    color_idx = (i // 2) % len(label_colors)  # Pair-wise color assignment
    ax1.text(x[i], -0.25, label, ha='center', va='top', fontsize=10, rotation=90, color=label_colors[color_idx])

# Adjust plot limits to accommodate rotated labels
plt.subplots_adjust(bottom=0.35)

# Add grid and legend
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)

# Set background colors
ax1.set_facecolor('#f5f5f5')
fig.patch.set_facecolor('#fafafa')

# Finalize plot
plt.title('Comparison of Test Metrics Across Models with 95% CI', fontsize=14, pad=15)
plt.savefig('comparison_metrics.png', bbox_inches='tight')
plt.show()
