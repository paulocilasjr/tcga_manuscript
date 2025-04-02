import matplotlib.pyplot as plt
import numpy as np
import json
import sys

# Function to load JSON data
def load_json_data(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{filename}' is not a valid JSON file")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        sys.exit(1)

# Check if filename is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <json_file>")
    sys.exit(1)

filename = sys.argv[1]
data = load_json_data(filename)

# Extract data from JSON
results = [entry['model'] for entry in data]
metrics_dict = {
    'loss': [entry['loss']['value'] for entry in data],
    'accuracy': [entry['accuracy']['value'] for entry in data],
    'precision': [entry['precision']['value'] for entry in data],
    'recall': [entry['recall']['value'] for entry in data],
    'roc_auc': [entry['roc_auc']['value'] for entry in data],
    'specificity': [entry['specificity']['value'] for entry in data],
    'f1_score': [entry['f1_score']['value'] for entry in data]
}
ci_lower_dict = {
    'loss': [entry['loss']['ci_lower'] for entry in data],
    'accuracy': [entry['accuracy']['ci_lower'] for entry in data],
    'precision': [entry['precision']['ci_lower'] for entry in data],
    'recall': [entry['recall']['ci_lower'] for entry in data],
    'roc_auc': [entry['roc_auc']['ci_lower'] for entry in data],
    'specificity': [entry['specificity']['ci_lower'] for entry in data],
    'f1_score': [entry['f1_score']['ci_lower'] for entry in data]
}
ci_upper_dict = {
    'loss': [entry['loss']['ci_upper'] for entry in data],
    'accuracy': [entry['accuracy']['ci_upper'] for entry in data],
    'precision': [entry['precision']['ci_upper'] for entry in data],
    'recall': [entry['recall']['ci_upper'] for entry in data],
    'roc_auc': [entry['roc_auc']['ci_upper'] for entry in data],
    'specificity': [entry['specificity']['ci_upper'] for entry in data],
    'f1_score': [entry['f1_score']['ci_upper'] for entry in data]
}

# Define metrics and names
metrics = [metrics_dict['accuracy'], metrics_dict['precision'], metrics_dict['recall'], 
           metrics_dict['roc_auc'], metrics_dict['specificity'], metrics_dict['f1_score']]
metric_names = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'Specificity', 'F1-Score']
ci_lower = [ci_lower_dict['accuracy'], ci_lower_dict['precision'], ci_lower_dict['recall'], 
            ci_lower_dict['roc_auc'], ci_lower_dict['specificity'], ci_lower_dict['f1_score']]
ci_upper = [ci_upper_dict['accuracy'], ci_upper_dict['precision'], ci_upper_dict['recall'], 
            ci_upper_dict['roc_auc'], ci_upper_dict['specificity'], ci_upper_dict['f1_score']]
loss = metrics_dict['loss']
loss_ci_lower = ci_lower_dict['loss']
loss_ci_upper = ci_upper_dict['loss']

# Colors for plotting
colors = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F']
loss_color = '#B3B3B3'

# Create the plot
fig, ax1 = plt.subplots(figsize=(20, 8))
ax2 = ax1.twinx()

bar_width = 0.12
x = np.arange(len(results))
n_metrics = len(metrics)

# Plot performance metrics
for i, (metric, name, color, lower, upper) in enumerate(zip(metrics, metric_names, colors, ci_lower, ci_upper)):
    offset = (i - n_metrics / 2) * bar_width
    bars = ax1.bar(x + offset, metric, bar_width, label=name, color=color, edgecolor='white', linewidth=0.5)
    yerr_lower = [m - l if l is not None else 0 for m, l in zip(metric, lower)]
    yerr_upper = [u - m if u is not None else 0 for m, u in zip(metric, upper)]
    yerr = np.array([yerr_lower, yerr_upper])
    ax1.errorbar(x + offset, metric, yerr=yerr, fmt='none', color='black', capsize=3, linewidth=1)

# Plot loss
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
ax1.set_xticklabels(results, fontsize=10, rotation=45, ha='right')
ax1.set_ylim(0, 1.1)
ax2.set_ylim(0, max(loss) * 1.2)  # Dynamically adjust loss y-axis based on max value

ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)

# Styling
ax1.set_facecolor('#f5f5f5')
fig.patch.set_facecolor('#fafafa')

plt.tight_layout()
plt.title('Comparison of Metrics Across Models with 95% CI', fontsize=14, pad=15)
plt.savefig('model_metrics_results.png', dpi=300, bbox_inches='tight')
plt.show()
