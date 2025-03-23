import matplotlib.pyplot as plt
import numpy as np

# New results list (flattened to include Train, Val, Test for each model/variant)
results = [
    'Pytorch ResNet50 MILc Leak Train', 'Pytorch ResNet50 MILc Leak Val', 'Pytorch ResNet50 MILc Leak Test',
    'Pytorch ResNet50 MILc No Leak Train', 'Pytorch ResNet50 MILc No Leak Val', 'Pytorch ResNet50 MILc No Leak Test',
    'Pytorch ResNet50 MILc Seeds Train', 'Pytorch ResNet50 MILc Seeds Val', 'Pytorch ResNet50 MILc Seeds Test',
    'Pytorch ResNet50 MILf Leak Train', 'Pytorch ResNet50 MILf Leak Val', 'Pytorch ResNet50 MILf Leak Test',
    'Pytorch ResNet50 MILf No Leak Train', 'Pytorch ResNet50 MILf No Leak Val', 'Pytorch ResNet50 MILf No Leak Test',
    'Pytorch ResNet50 MILf Seeds Train', 'Pytorch ResNet50 MILf Seeds Val', 'Pytorch ResNet50 MILf Seeds Test',
    'Pytorch VGG11 MILc Leak Train', 'Pytorch VGG11 MILc Leak Val', 'Pytorch VGG11 MILc Leak Test',
    'Pytorch VGG11 MILc No Leak Train', 'Pytorch VGG11 MILc No Leak Val', 'Pytorch VGG11 MILc No Leak Test',
    'Pytorch VGG11 MILc Seeds Train', 'Pytorch VGG11 MILc Seeds Val', 'Pytorch VGG11 MILc Seeds Test',
    'Ludwig ResNet50 MILc Train', 'Ludwig ResNet50 MILc Val', 'Ludwig ResNet50 MILc Test',
    'Ludwig ResNet50 MILf Train', 'Ludwig ResNet50 MILf Val', 'Ludwig ResNet50 MILf Test',
    'Ludwig VGG11 MILc Train', 'Ludwig VGG11 MILc Val', 'Ludwig VGG11 MILc Test'
]

# Metric arrays (means and confidence intervals where available)
loss = [
    0.0655, 0.1266, 0.1270, 0.1115, 0.5452, 0.5963, 0.1379, 0.5414, 0.5745,
    0.0655, 0.1266, 0.1270, 0.0808, 0.2714, 0.2751, 0.0854, 0.2659, 0.2701,
    0.0171, 0.3609, 0.3683, 0.5573, 0.6064, 0.6956, 0.1685, 0.4059, 0.8189,
    0.4047, 0.5937, 0.5751, 0.1670, 0.4029, 0.4268, 1.2731, 1.3587, 3.7357
]
loss_ci_lower = [
    0.0529, 0.1215, 0.1218, None, None, None, 0.1291, 0.5358, 0.5661,
    0.0529, 0.1215, 0.1218, None, None, None, 0.0798, 0.2612, 0.2657,
    0.0056, 0.3160, 0.3235, None, None, None, 0.1053, 0.3906, 0.7532,
    None, None, None, None, None, None, None, None, None
]
loss_ci_upper = [
    0.0782, 0.1316, 0.1322, None, None, None, 0.1468, 0.5469, 0.5828,
    0.0782, 0.1316, 0.1322, None, None, None, 0.0910, 0.2706, 0.2745,
    0.0287, 0.4057, 0.4130, None, None, None, 0.2317, 0.4213, 0.8846,
    None, None, None, None, None, None, None, None, None
]

accuracy = [
    0.9816, 0.9456, 0.9448, 0.9790, 0.6879, 0.6711, 0.9675, 0.7035, 0.6777,
    0.9816, 0.9456, 0.9448, 0.9754, 0.8720, 0.8793, 0.9734, 0.8752, 0.8797,
    1.0000, 0.8780, 0.8698, 0.8686, 0.8189, 0.5166, 0.9557, 0.8071, 0.5693,
    0.8161, 0.6773, 0.7164, 0.9539, 0.8864, 0.8909, 0.7338, 0.7143, 0.5656
]
accuracy_ci_lower = [
    0.9765, 0.9431, 0.9420, None, None, None, 0.9637, 0.6962, 0.6723,
    0.9765, 0.9431, 0.9420, None, None, None, 0.9711, 0.8728, 0.8786,
    1.0000, 0.8697, 0.8600, None, None, None, 0.9255, 0.7950, 0.5627,
    None, None, None, None, None, None, None, None, None
]
accuracy_ci_upper = [
    0.9867, 0.9481, 0.9475, None, None, None, 0.9714, 0.7109, 0.6830,
    0.9867, 0.9481, 0.9475, None, None, None, 0.9757, 0.8776, 0.8808,
    1.0000, 0.8862, 0.8795, None, None, None, 0.9860, 0.8193, 0.5758,
    None, None, None, None, None, None, None, None, None
]

precision = [
    0.9960, 0.9747, 0.9736, 0.9994, 0.7005, 0.6918, 0.9992, 0.7202, 0.6986,
    0.9960, 0.9747, 0.9736, 0.9950, 0.8834, 0.8840, 0.9944, 0.8902, 0.8860,
    1.0000, 0.9758, 0.9701, 1.0000, 0.8814, 0.5833, 1.0000, 0.8331, 0.5844,
    0.8402, 0.8571, 0.8125, 0.9792, 0.9160, 0.9029, 0.8086, 0.7054, 0.5678
]
precision_ci_lower = [
    0.9944, 0.9726, 0.9705, None, None, None, 0.9987, 0.7108, 0.6933,
    0.9944, 0.9726, 0.9705, None, None, None, 0.9936, 0.8869, 0.8847,
    1.0000, 0.9698, 0.9643, None, None, None, 1.0000, 0.8094, 0.5700,
    None, None, None, None, None, None, None, None, None
]
precision_ci_upper = [
    0.9975, 0.9767, 0.9768, None, None, None, 0.9997, 0.7296, 0.7038,
    0.9975, 0.9767, 0.9768, None, None, None, 0.9952, 0.8935, 0.8873,
    1.0000, 0.9818, 0.9759, None, None, None, 1.0000, 0.8569, 0.5988,
    None, None, None, None, None, None, None, None, None
]

recall = [
    0.9823, 0.9611, 0.9612, 0.9699, 0.8480, 0.8946, 0.9532, 0.8368, 0.8907,
    0.9823, 0.9611, 0.9612, 0.9762, 0.9705, 0.9883, 0.9744, 0.9655, 0.9859,
    1.0000, 0.8381, 0.8306, 0.8092, 0.8447, 0.5292, 0.9357, 0.8955, 0.8667,
    0.9023, 0.5614, 0.7352, 0.9668, 0.9470, 0.9773, 0.8035, 0.9886, 0.9931
]
recall_ci_lower = [
    0.9778, 0.9594, 0.9588, None, None, None, 0.9476, 0.8265, 0.8828,
    0.9778, 0.9594, 0.9588, None, None, None, 0.9724, 0.9638, 0.9850,
    1.0000, 0.8244, 0.8185, None, None, None, 0.8917, 0.8780, 0.7931,
    None, None, None, None, None, None, None, None, None
]
recall_ci_upper = [
    0.9868, 0.9629, 0.9637, None, None, None, 0.9589, 0.8472, 0.8987,
    0.9868, 0.9629, 0.9637, None, None, None, 0.9764, 0.9672, 0.9868,
    1.0000, 0.8519, 0.8426, None, None, None, 0.9796, 0.9129, 0.9402,
    None, None, None, None, None, None, None, None, None
]

roc_auc = [
    0.9957, 0.9801, 0.9799, 0.9993, 0.7990, 0.7518, 0.9980, 0.7934, 0.7598,
    0.9957, 0.9801, 0.9799, 0.9940, 0.9306, 0.9197, 0.9934, 0.9317, 0.9169,
    1.0000, 0.9393, 0.9348, 0.9801, 0.9044, 0.5130, 0.9983, 0.8938, 0.5425,
    0.8824, 0.8114, 0.7748, 0.9847, 0.9335, 0.9374, 0.7793, 0.8786, 0.5022
]
roc_auc_ci_lower = [
    0.9942, 0.9788, 0.9782, None, None, None, 0.9974, 0.7894, 0.7550,
    0.9942, 0.9788, 0.9782, None, None, None, 0.9925, 0.9293, 0.9148,
    1.0000, 0.9320, 0.9269, None, None, None, 0.9962, 0.8927, 0.5358,
    None, None, None, None, None, None, None, None, None
]
roc_auc_ci_upper = [
    0.9972, 0.9815, 0.9817, None, None, None, 0.9986, 0.7973, 0.7646,
    0.9972, 0.9815, 0.9817, None, None, None, 0.9942, 0.9341, 0.9190,
    1.0000, 0.9467, 0.9427, None, None, None, 1.0003, 0.8948, 0.5491,
    None, None, None, None, None, None, None, None, None
]

specificity = [
    0.9771, 0.8562, 0.8500, 0.9987, 0.4414, 0.2512, 0.9984, 0.4982, 0.2773,
    0.9771, 0.8562, 0.8500, 0.9708, 0.4455, 0.2455, 0.9673, 0.4844, 0.2617,
    1.0000, 0.9579, 0.9485, 1.0000, 0.7656, 0.5000, 1.0000, 0.6250, 0.1759,
    0.6302, 0.8559, 0.6812, 0.8767, 0.6241, 0.3885, 0.5798, 0.1484, 0.0000
]
specificity_ci_lower = [
    0.9684, 0.8441, 0.8313, None, None, None, 0.9974, 0.4707, 0.2558,
    0.9684, 0.8441, 0.8313, None, None, None, 0.9627, 0.4665, 0.2517,
    1.0000, 0.9471, 0.9385, None, None, None, 1.0000, 0.5562, 0.0659,
    None, None, None, None, None, None, None, None, None
]
specificity_ci_upper = [
    0.9859, 0.8683, 0.8688, None, None, None, 0.9994, 0.5257, 0.2988,
    0.9859, 0.8683, 0.8688, None, None, None, 0.9719, 0.5023, 0.2717,
    1.0000, 0.9688, 0.9586, None, None, None, 1.0000, 0.6938, 0.2859,
    None, None, None, None, None, None, None, None, None
]

f1_score = [
    0.9891, 0.9678, 0.9674, 0.9844, 0.7672, 0.7803, 0.9757, 0.7740, 0.7830,
    0.9891, 0.9678, 0.9674, 0.9855, 0.9249, 0.9332, 0.9843, 0.9263, 0.9333,
    1.0000, 0.9016, 0.8948, 0.8945, 0.8627, 0.5550, 0.9658, 0.8623, 0.6943,
    0.8702, 0.6784, 0.7718, 0.9730, 0.9313, 0.9386, 0.8062, 0.8234, 0.7224
]
f1_score_ci_lower = [
    0.9861, 0.9664, 0.9658, None, None, None, 0.9727, 0.7696, 0.7798,
    0.9861, 0.9664, 0.9658, None, None, None, 0.9830, 0.9250, 0.9327,
    1.0000, 0.8942, 0.8865, None, None, None, 0.9419, 0.8564, 0.6773,
    None, None, None, None, None, None, None, None, None
]
f1_score_ci_upper = [
    0.9921, 0.9693, 0.9690, None, None, None, 0.9786, 0.7783, 0.7861,
    0.9921, 0.9693, 0.9690, None, None, None, 0.9857, 0.9276, 0.9338,
    1.0000, 0.9090, 0.9031, None, None, None, 0.9897, 0.8683, 0.7114,
    None, None, None, None, None, None, None, None, None
]

metrics = [accuracy, precision, recall, roc_auc, specificity, f1_score]
metric_names = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'Specificity', 'F1-Score']
ci_lower = [accuracy_ci_lower, precision_ci_lower, recall_ci_lower, roc_auc_ci_lower, specificity_ci_lower, f1_score_ci_lower]
ci_upper = [accuracy_ci_upper, precision_ci_upper, recall_ci_upper, roc_auc_ci_upper, specificity_ci_upper, f1_score_ci_upper]

colors = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F']
loss_color = '#B3B3B3'

fig, ax1 = plt.subplots(figsize=(20, 8))  # Increased figure size for more entries
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

ax1.set_xlabel('Models and Splits', fontsize=12)
ax1.set_ylabel('Performance Metrics Evaluation', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(results, fontsize=10, rotation=45, ha='right')  # Adjusted rotation for readability
ax1.set_ylim(0, 1.1)  # Adjusted ylim to accommodate higher values
ax2.set_ylim(0, 4)    # Adjusted ylim for higher loss values (e.g., Ludwig VGG11)

ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)

ax1.set_facecolor('#f5f5f5')
fig.patch.set_facecolor('#fafafa')

plt.tight_layout()
plt.title('Comparison of Metrics Across Models and Splits with 95% CI', fontsize=14, pad=15)
plt.savefig('MIL_metrics_results.png', dpi=300, bbox_inches='tight')  # Higher DPI for clarity
plt.show()
