import matplotlib.pyplot as plt
import numpy as np

# Updated results with new order
results = [
    'Pytorch UNIv2 cross-validation',
    'Ludwig UNIv2',
    'Pytorch Resnet50_f cross-validation',
    'Ludwig Resnet50_f',
    'Pytorch Resnet50_c cross-validation',
    'Ludwig Resnet50_c',
    'Ludwig VGG11_f',
    'Pytorch VGG11 cross-validation',
    'Ludwig VGG11'
]

# Define the new order index mapping
new_order = [3, 7, 1, 5, 0, 4, 8, 2, 6]

def reorder_list(lst, order):
    return [lst[i] for i in order]

# Reorder all metric lists
loss = reorder_list([1.0785, 1.9936, 0.5503, 0.6007, 0.8014, 1.0982, 1.1385, 0.9623, 1.1385], new_order)
loss_ci_lower = reorder_list([0.8674, None, 0.5208, None, 0.6932, None, None, 0.7394, None], new_order)
loss_ci_upper = reorder_list([1.2895, None, 0.5799, None, 0.9473, None, None, 2.9727, None], new_order)

accuracy = reorder_list([0.7592, 0.7398, 0.7613, 0.7532, 0.5822, 0.5924, 0.5377, 0.5814, 0.6100], new_order)
accuracy_ci_lower = reorder_list([0.7424, None, 0.7361, None, 0.5669, None, None, 0.5550, None], new_order)
accuracy_ci_upper = reorder_list([0.7760, None, 0.7865, None, 0.6051, None, None, 0.6281, None], new_order)

precision = reorder_list([0.8056, 0.7850, 0.7613, 0.8082, 0.5769, 0.6096, 0.7585, 0.5734, 0.6296], new_order)
precision_ci_lower = reorder_list([0.7907, None, 0.7361, None, 0.4560, None, None, 0.4608, None], new_order)
precision_ci_upper = reorder_list([0.8206, None, 0.7865, None, 0.6459, None, None, 0.6939, None], new_order)

recall = reorder_list([0.9019, 0.8859, 1.0000, 0.8931, 0.5558, 0.5039, 0.7953, 0.5450, 0.5258], new_order)
recall_ci_lower = reorder_list([0.8856, None, 1.0000, None, 0.4591, None, None, 0.4739, None], new_order)
recall_ci_upper = reorder_list([0.9182, None, 1.0000, None, 0.5986, None, None, 0.5938, None], new_order)

roc_auc = reorder_list([0.7057, 0.6930, 0.5000, 0.6571, 0.6238, 0.6435, 0.9264, 0.6241, 0.6597], new_order)
roc_auc_ci_lower = reorder_list([0.6625, None, 0.5000, None, 0.5797, None, None, 0.5911, None], new_order)
roc_auc_ci_upper = reorder_list([0.7490, None, 0.5000, None, 0.6579, None, None, 0.7104, None], new_order)

specificity = reorder_list([0.2996, 0.3454, 0.0000, 0.2739, 0.6164, 0.6801, 0.6510, 0.6228, 0.6934], new_order)
specificity_ci_lower = reorder_list([0.2537, None, 0.0000, None, 0.5913, None, None, 0.5613, None], new_order)
specificity_ci_upper = reorder_list([0.3455, None, 0.0000, None, 0.6501, None, None, 0.6998, None], new_order)

f1_score = reorder_list([0.8508, 0.8325, 0.8640, 0.8486, 0.5703, 0.5530, 0.5377, 0.5544, 0.5732], new_order)
f1_score_ci_lower = reorder_list([0.8397, None, 0.8477, None, 0.4591, None, None, 0.4896, None], new_order)
f1_score_ci_upper = reorder_list([0.8619, None, 0.8804, None, 0.6166, None, None, 0.6214, None], new_order)

metrics = [accuracy, precision, recall, roc_auc, specificity, f1_score]
metric_names = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'Specificity', 'F1-Score']
ci_lower = [accuracy_ci_lower, precision_ci_lower, recall_ci_lower, roc_auc_ci_lower, specificity_ci_lower, f1_score_ci_lower]
ci_upper = [accuracy_ci_upper, precision_ci_upper, recall_ci_upper, roc_auc_ci_upper, specificity_ci_upper, f1_score_ci_upper]

colors = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F']
loss_color = '#B3B3B3'

fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

bar_width = 0.12
x = np.arange(len(results))
n_metrics = len(metrics)

for i, (metric, name, color, lower, upper) in enumerate(zip(metrics, metric_names, colors, ci_lower, ci_upper)):
    offset = (i - n_metrics / 2) * bar_width
    bars = ax1.bar(x + offset, metric, bar_width, label=name, color=color, edgecolor='white', linewidth=0.5)
    yerr_lower = [m - l if l is not None else 0 for m, l in zip(metric, lower)]
    yerr_upper = [u - m if u is not None else 0 for m, u in zip(metric, upper)]
    yerr = np.array([yerr_lower, yerr_upper])
    ax1.errorbar(x + offset, metric, yerr=yerr, fmt='none', color='black', capsize=3, linewidth=1)

loss_offset = (n_metrics / 2) * bar_width + bar_width * 0.5
bars = ax2.bar(x + loss_offset, loss, bar_width, label='Loss', color=loss_color, edgecolor='white', linewidth=0.5)
yerr_lower = [l - ll if ll is not None else 0 for l, ll in zip(loss, loss_ci_lower)]
yerr_upper = [lu - l if lu is not None else 0 for l, lu in zip(loss, loss_ci_upper)]
yerr = np.array([yerr_lower, yerr_upper])
ax2.errorbar(x + loss_offset, loss, yerr=yerr, fmt='none', color='black', capsize=3, linewidth=1)

# Add a thin red dotted line at y=0.7 on ax1
ax1.axhline(y=0.7, color='red', linestyle=':', linewidth=1, label='Threshold 0.7')

ax1.set_xlabel('Models', fontsize=12)
ax1.set_ylabel('Performance Metrics Evaluation', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax1.set_xticks(x)
# Modified xticklabels with line breaks and 90-degree rotation
ax1.set_xticklabels([label.replace(' ', '\n') for label in results], fontsize=10, rotation=90)
ax1.set_ylim(0, 1.1)
ax2.set_ylim(0, 3)

ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)

ax1.set_facecolor('#f5f5f5')
fig.patch.set_facecolor('#fafafa')

plt.tight_layout()
plt.title('Comparison of Test Metrics Across Models with 95% CI', fontsize=14, pad=15)
plt.savefig('comparison_metrics.png')
plt.show()
