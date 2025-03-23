import matplotlib.pyplot as plt
import numpy as np

# Updated results list with only Test splits
results = [
    'Pytorch ResNet50 MILc Leak Test',
    'Pytorch ResNet50 MILc No Leak Test',
    'Pytorch ResNet50 MILc Seeds Test',
    'Pytorch ResNet50 MILf Leak Test',
    'Pytorch ResNet50 MILf No Leak Test',
    'Pytorch ResNet50 MILf Seeds Test',
    'Pytorch VGG11 MILc Leak Test',
    'Pytorch VGG11 MILc No Leak Test',
    'Pytorch VGG11 MILc Seeds Test',
    'Ludwig ResNet50 MILc Test',
    'Ludwig ResNet50 MILf Test',
    'Ludwig VGG11 MILc Test'
]

# Metric arrays (only Test values)
loss = [
    0.1270, 0.5963, 0.5745, 0.1270, 0.2751, 0.2701,
    0.3683, 0.6956, 0.8189, 0.5751, 0.4268, 3.7357
]
loss_ci_lower = [
    0.1218, None, 0.5661, 0.1218, None, 0.2657,
    0.3235, None, 0.7532, None, None, None
]
loss_ci_upper = [
    0.1322, None, 0.5828, 0.1322, None, 0.2745,
    0.4130, None, 0.8846, None, None, None
]

accuracy = [
    0.9448, 0.6711, 0.6777, 0.9448, 0.8793, 0.8797,
    0.8698, 0.5166, 0.5693, 0.7164, 0.8909, 0.5656
]
accuracy_ci_lower = [
    0.9420, None, 0.6723, 0.9420, None, 0.8786,
    0.8600, None, 0.5627, None, None, None
]
accuracy_ci_upper = [
    0.9475, None, 0.6830, 0.9475, None, 0.8808,
    0.8795, None, 0.5758, None, None, None
]

precision = [
    0.9736, 0.6918, 0.6986, 0.9736, 0.8840, 0.8860,
    0.9701, 0.5833, 0.5844, 0.8125, 0.9029, 0.5678
]
precision_ci_lower = [
    0.9705, None, 0.6933, 0.9705, None, 0.8847,
    0.9643, None, 0.5700, None, None, None
]
precision_ci_upper = [
    0.9768, None, 0.7038, 0.9768, None, 0.8873,
    0.9759, None, 0.5988, None, None, None
]

recall = [
    0.9612, 0.8946, 0.8907, 0.9612, 0.9883, 0.9859,
    0.8306, 0.5292, 0.8667, 0.7352, 0.9773, 0.9931
]
recall_ci_lower = [
    0.9588, None, 0.8828, 0.9588, None, 0.9850,
    0.8185, None, 0.7931, None, None, None
]
recall_ci_upper = [
    0.9637, None, 0.8987, 0.9637, None, 0.9868,
    0.8426, None, 0.9402, None, None, None
]

roc_auc = [
    0.9799, 0.7518, 0.7598, 0.9799, 0.9197, 0.9169,
    0.9348, 0.5130, 0.5425, 0.7748, 0.9374, 0.5022
]
roc_auc_ci_lower = [
    0.9782, None, 0.7550, 0.9782, None, 0.9148,
    0.9269, None, 0.5358, None, None, None
]
roc_auc_ci_upper = [
    0.9817, None, 0.7646, 0.9817, None, 0.9190,
    0.9427, None, 0.5491, None, None, None
]

specificity = [
    0.8500, 0.2512, 0.2773, 0.8500, 0.2455, 0.2617,
    0.9485, 0.5000, 0.1759, 0.6812, 0.3885, 0.0000
]
specificity_ci_lower = [
    0.8313, None, 0.2558, 0.8313, None, 0.2517,
    0.9385, None, 0.0659, None, None, None
]
specificity_ci_upper = [
    0.8688, None, 0.2988, 0.8688, None, 0.2717,
    0.9586, None, 0.2859, None, None, None
]

f1_score = [
    0.9674, 0.7803, 0.7830, 0.9674, 0.9332, 0.9333,
    0.8948, 0.5550, 0.6943, 0.7718, 0.9386, 0.7224
]
f1_score_ci_lower = [
    0.9658, None, 0.7798, 0.9658, None, 0.9327,
    0.8865, None, 0.6773, None, None, None
]
f1_score_ci_upper = [
    0.9690, None, 0.7861, 0.9690, None, 0.9338,
    0.9031, None, 0.7114, None, None, None
]

metrics = [accuracy, precision, recall, roc_auc, specificity, f1_score]
metric_names = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'Specificity', 'F1-Score']
ci_lower = [accuracy_ci_lower, precision_ci_lower, recall_ci_lower, roc_auc_ci_lower, specificity_ci_lower, f1_score_ci_lower]
ci_upper = [accuracy_ci_upper, precision_ci_upper, recall_ci_upper, roc_auc_ci_upper, specificity_ci_upper, f1_score_ci_upper]

colors = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F']
loss_color = '#B3B3B3'

fig, ax1 = plt.subplots(figsize=(14, 6))  # Adjusted figure size back to original
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

ax1.set_xlabel('Models (Test Split)', fontsize=12)
ax1.set_ylabel('Performance Metrics Evaluation', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(results, fontsize=10, rotation=15)  # Reduced rotation angle
ax1.set_ylim(0, 1.1)  # Kept ylim for consistency
ax2.set_ylim(0, 4)    # Kept ylim to accommodate Ludwig VGG11 MILc Test loss

ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)

ax1.set_facecolor('#f5f5f5')
fig.patch.set_facecolor('#fafafa')

plt.tight_layout()
plt.title('Comparison of Test Metrics Across Models with 95% CI', fontsize=14, pad=15)
plt.savefig('MIL_test_metrics.png', dpi=300, bbox_inches='tight')
plt.show()
