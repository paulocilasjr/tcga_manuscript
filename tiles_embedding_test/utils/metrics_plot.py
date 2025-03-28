import matplotlib.pyplot as plt
import numpy as np

# Updated results with new entries
results = [
    'Pytorch Resnet50_c cross-validation',
    'Pytorch Resnet50_f cross-validation',
    'Pytorch VGG11 cross-validation',
    'Pytorch UNIv2 cross-validation',
    'Ludwig Resnet50_c',
    'Ludwig Resnet50_f',
    'Ludwig VGG11',
    'Ludwig UNIv2'
]

# Metric arrays updated with new values
loss = [0.8014, 0.5503, 0.9623, 1.0785, 1.0982, 0.6007, 1.1385, 1.9936]
loss_ci_lower = [0.6932, 0.5208, 0.7394, 0.8674, None, None, None, None]
loss_ci_upper = [0.9473, 0.5799, 2.9727, 1.2895, None, None, None, None]

accuracy = [0.5822, 0.7613, 0.5814, 0.7592, 0.5924, 0.7532, 0.6100, 0.7398]
accuracy_ci_lower = [0.5669, 0.7361, 0.5550, 0.7424, None, None, None, None]
accuracy_ci_upper = [0.6051, 0.7865, 0.6281, 0.7760, None, None, None, None]

precision = [0.5769, 0.7613, 0.5734, 0.8056, 0.6096, 0.8082, 0.6296, 0.7850]
precision_ci_lower = [0.4560, 0.7361, 0.4608, 0.7907, None, None, None, None]
precision_ci_upper = [0.6459, 0.7865, 0.6939, 0.8206, None, None, None, None]

recall = [0.5558, 1.0000, 0.5450, 0.9019, 0.5039, 0.8931, 0.5258, 0.8859]
recall_ci_lower = [0.4591, 1.0000, 0.4739, 0.8856, None, None, None, None]
recall_ci_upper = [0.5986, 1.0000, 0.5938, 0.9182, None, None, None, None]

roc_auc = [0.6238, 0.5000, 0.6241, 0.7057, 0.6435, 0.6571, 0.6597, 0.6930]
roc_auc_ci_lower = [0.5797, 0.5000, 0.5911, 0.6625, None, None, None, None]
roc_auc_ci_upper = [0.6579, 0.5000, 0.7104, 0.7490, None, None, None, None]

specificity = [0.6164, 0.0000, 0.6228, 0.2996, 0.6801, 0.2739, 0.6934, 0.3454]
specificity_ci_lower = [0.5913, 0.0000, 0.5613, 0.2537, None, None, None, None]
specificity_ci_upper = [0.6501, 0.0000, 0.6998, 0.3455, None, None, None, None]

f1_score = [0.5703, 0.8640, 0.5544, 0.8508, 0.5530, 0.8486, 0.5732, 0.8325]
f1_score_ci_lower = [0.4591, 0.8477, 0.4896, 0.8397, None, None, None, None]
f1_score_ci_upper = [0.6166, 0.8804, 0.6214, 0.8619, None, None, None, None]

metrics = [accuracy, precision, recall, roc_auc, specificity, f1_score]
metric_names = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'Specificity', 'F1-Score']
ci_lower = [accuracy_ci_lower, precision_ci_lower, recall_ci_lower, roc_auc_ci_lower, specificity_ci_lower, f1_score_ci_lower]
ci_upper = [accuracy_ci_upper, precision_ci_upper, recall_ci_upper, roc_auc_ci_upper, specificity_ci_upper, f1_score_ci_upper]

colors = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F']
loss_color = '#B3B3B3'

fig, ax1 = plt.subplots(figsize=(6, 14))  # Swapped dimensions for vertical orientation
ax2 = ax1.twiny()  # Changed to twiny for horizontal bars

bar_height = 0.12  # Changed to bar_height
y = np.arange(len(results))  # Changed x to y
n_metrics = len(metrics)

# Plot horizontal bars instead of vertical
for i, (metric, name, color, lower, upper) in enumerate(zip(metrics, metric_names, colors, ci_lower, ci_upper)):
    offset = (i - n_metrics / 2) * bar_height
    bars = ax1.barh(y + offset, metric, bar_height, label=name, color=color, edgecolor='white', linewidth=0.5)
    xerr_lower = [m - l if l is not None else 0 for m, l in zip(metric, lower)]
    xerr_upper = [u - m if u is not None else 0 for m, u in zip(metric, upper)]
    xerr = np.array([xerr_lower, xerr_upper])
    ax1.errorbar(metric, y + offset, xerr=xerr, fmt='none', color='black', capsize=3, linewidth=1)

loss_offset = (n_metrics / 2) * bar_height + bar_height * 0.5
bars = ax2.barh(y + loss_offset, loss, bar_height, label='Loss', color=loss_color, edgecolor='white', linewidth=0.5)
xerr_lower = [l - ll if ll is not None else 0 for l, ll in zip(loss, loss_ci_lower)]
xerr_upper = [lu - l if lu is not None else 0 for l, lu in zip(loss, loss_ci_upper)]
xerr = np.array([xerr_lower, xerr_upper])
ax2.errorbar(loss, y + loss_offset, xerr=xerr, fmt='none', color='black', capsize=3, linewidth=1)

# Add a thin red dotted line at x=0.7 on ax1 (changed from y to x)
ax1.axvline(x=0.7, color='red', linestyle=':', linewidth=1, label='Threshold 0.7')

ax1.set_ylabel('Models', fontsize=12)
ax1.set_xlabel('Performance Metrics Evaluation', fontsize=12)
ax2.set_xlabel('Loss', fontsize=12)
ax1.set_yticks(y)
# Modified xticklabels to allow line breaks
ax1.set_yticklabels([label.replace(' ', '\n') for label in results], fontsize=10)
ax1.set_xlim(0, 1.1)  # Changed ylim to xlim
ax2.set_xlim(0, 3)    # Changed ylim to xlim

ax1.grid(True, axis='x', linestyle='--', alpha=0.7)  # Changed axis from 'y' to 'x'

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)

ax1.set_facecolor('#f5f5f5')
fig.patch.set_facecolor('#fafafa')

plt.tight_layout()
plt.title('Comparison of Test Metrics Across Models with 95% CI', fontsize=14, pad=15)
plt.savefig('comparison_metrics.png')
plt.show()
