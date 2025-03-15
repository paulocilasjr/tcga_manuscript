import matplotlib.pyplot as plt
import numpy as np

# Data from the results (Test set values only, with CIs where available)
results = ['Pytorch Resnet50 cross-validation', 'Pytorch VGG11 cross-validation', 'Ludwig Resnet50', 'Ludwig VGG11']

# Loss with CIs for Result 1 and 2
loss = [0.8014, 0.9623, 1.0982, 1.1385]
loss_ci_lower = [0.6932, 0.7394, None, None]  # Lower bounds for Result 1 and 2
loss_ci_upper = [0.9473, 2.9727, None, None]  # Upper bounds for Result 1 and 2

# Accuracy with CIs for Result 1 and 2
accuracy = [0.5822, 0.5814, 0.5924, 0.6100]
accuracy_ci_lower = [0.5669, 0.5550, None, None]
accuracy_ci_upper = [0.6051, 0.6281, None, None]

# Precision with CIs for Result 1 and 2
precision = [0.5769, 0.5734, 0.6096, 0.6296]
precision_ci_lower = [0.4560, 0.4608, None, None]
precision_ci_upper = [0.6459, 0.6939, None, None]

# Recall with CIs for Result 1 and 2
recall = [0.5558, 0.5450, 0.5039, 0.5258]
recall_ci_lower = [0.4591, 0.4739, None, None]
recall_ci_upper = [0.5986, 0.5938, None, None]

# ROC-AUC with CIs for Result 1 and 2
roc_auc = [0.6238, 0.6241, 0.6435, 0.6597]
roc_auc_ci_lower = [0.5797, 0.5911, None, None]
roc_auc_ci_upper = [0.6579, 0.7104, None, None]

# Specificity with CIs for Result 1 and 2
specificity = [0.6164, 0.6228, 0.6801, 0.6934]
specificity_ci_lower = [0.5913, 0.5613, None, None]
specificity_ci_upper = [0.6501, 0.6998, None, None]

# F1-Score with CIs for Result 1 and 2
f1_score = [0.5703, 0.5544, 0.5530, 0.5732]
f1_score_ci_lower = [0.4591, 0.4896, None, None]
f1_score_ci_upper = [0.6166, 0.6214, None, None]

# Combine all metrics except Loss for the left y-axis
metrics = [accuracy, precision, recall, roc_auc, specificity, f1_score]
metric_names = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'Specificity', 'F1-Score']
ci_lower = [accuracy_ci_lower, precision_ci_lower, recall_ci_lower, roc_auc_ci_lower, specificity_ci_lower, f1_score_ci_lower]
ci_upper = [accuracy_ci_upper, precision_ci_upper, recall_ci_upper, roc_auc_ci_upper, specificity_ci_upper, f1_score_ci_upper]

# ColorBrewer Set2 palette (colorblind-friendly)
colors = [
    '#66C2A5',  # Light Green (Accuracy)
    '#FC8D62',  # Soft Orange (Precision)
    '#8DA0CB',  # Light Blue (Recall)
    '#E78AC3',  # Pink (ROC-AUC)
    '#A6D854',  # Lime Green (Specificity)
    '#FFD92F',  # Yellow (F1-Score)
]
loss_color = '#B3B3B3'  # Gray (Loss)

# Set up the figure and axes
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

# Bar width and positioning
bar_width = 0.12
x = np.arange(len(results))
n_metrics = len(metrics)

# Plot bars for metrics (left y-axis) with error bars
for i, (metric, name, color, lower, upper) in enumerate(zip(metrics, metric_names, colors, ci_lower, ci_upper)):
    offset = (i - n_metrics / 2) * bar_width
    bars = ax1.bar(x + offset, metric, bar_width, label=name, color=color, edgecolor='white', linewidth=0.5)
    
    # Calculate error bar values (None for Results 3 and 4)
    yerr_lower = [m - l if l is not None else 0 for m, l in zip(metric, lower)]
    yerr_upper = [u - m if u is not None else 0 for m, u in zip(metric, upper)]
    yerr = np.array([yerr_lower, yerr_upper])
    
    # Add error bars
    ax1.errorbar(x + offset, metric, yerr=yerr, fmt='none', color='black', capsize=3, linewidth=1)

# Plot Loss bars (right y-axis) with error bars
loss_offset = (n_metrics / 2) * bar_width + bar_width * 0.5
bars = ax2.bar(x + loss_offset, loss, bar_width, label='Loss', color=loss_color, edgecolor='white', linewidth=0.5)
yerr_lower = [l - ll if ll is not None else 0 for l, ll in zip(loss, loss_ci_lower)]
yerr_upper = [lu - l if lu is not None else 0 for l, lu in zip(loss, loss_ci_upper)]
yerr = np.array([yerr_lower, yerr_upper])
ax2.errorbar(x + loss_offset, loss, yerr=yerr, fmt='none', color='black', capsize=3, linewidth=1)

# Customize the plot
ax1.set_xlabel('Models', fontsize=12)
ax1.set_ylabel('Performance Metrics Evaluation', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(results, fontsize=10, rotation=15)
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1.2)

# Add gridlines
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)

# Set a subtle background
ax1.set_facecolor('#f5f5f5')
fig.patch.set_facecolor('#fafafa')

# Adjust layout
plt.tight_layout()
plt.title('Comparison of Test Metrics Across Models with 95% CI', fontsize=14, pad=15)

plt.savefig('comparison_metrics.png')

# Show the plot
plt.show()

