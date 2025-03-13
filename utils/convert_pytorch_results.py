import numpy as np
import argparse

def analyze_cv_results(file_path):
    """
    Analyze and print cross-validation results from a specified file.
    
    Args:
        file_path (str): Full path to the input file (e.g., '1285.out' or '/path/to/1285.out')
    """
    # Temporary lists to store metric values
    metric_sets = {
        'Train': {'Loss': [], 'Accuracy': [], 'Precision': [], 'Recall': [],
                 'Roc_auc': [], 'Specificity': [], 'F1': []},
        'Validation': {'Loss': [], 'Accuracy': [], 'Precision': [], 'Recall': [],
                      'Roc_auc': [], 'Specificity': [], 'F1': []},
        'Test': {'Loss': [], 'Accuracy': [], 'Precision': [], 'Recall': [],
                'Roc_auc': [], 'Specificity': [], 'F1': []}
    }
    
    # Read and parse the file
    current_set = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if 'Fold' in line and 'Train Metrics' in line:
                    current_set = 'Train'
                elif 'Fold' in line and 'Validation Metrics' in line:
                    current_set = 'Validation'
                elif 'Fold' in line and 'Test Metrics' in line:
                    current_set = 'Test'
                elif line and current_set and ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        metric = parts[0].strip()
                        value = parts[-1].strip()
                        try:
                            value = float(value)
                            if metric in metric_sets[current_set]:
                                metric_sets[current_set][metric].append(value)
                        except ValueError:
                            continue  # Skip lines where conversion fails
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return

    # Calculate and print results directly
    for set_name, metrics in metric_sets.items():
        print(f"Cross-Validation Results ({set_name} Set):")
        print("Average Metrics:")
        for metric, values in metrics.items():
            if values:  # Only process if we have values
                median = np.median(values)
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                print(f"  {metric}: {median:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze cross-validation results from a file.")
    parser.add_argument("file_path", type=str, help="Path to the results file.")
    args = parser.parse_args()
    
    analyze_cv_results(args.file_path)

