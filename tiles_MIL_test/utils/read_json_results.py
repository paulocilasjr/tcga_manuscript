import json
import sys
import numpy as np
from scipy import stats

def calculate_stats(data):
    """Calculate mean and 95% CI for a list of numbers"""
    mean = np.mean(data)
    # Calculate 95% CI using t-distribution (updated for newer SciPy versions)
    ci = stats.t.interval(
        confidence=0.95,  # Changed from alpha to confidence
        df=len(data)-1,
        loc=mean,
        scale=stats.sem(data)
    )
    return mean, ci[0], ci[1]

def process_split(split_data, split_name):
    """Process metrics for a given split"""
    # Collect all metrics
    metrics = {}
    for entry in split_data:
        for metric, value in entry.items():
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append(value)
    
    # Calculate statistics for each metric
    print(f"\nResults for {split_name} split:")
    print("-" * 50)
    for metric in metrics:
        mean, ci_lower, ci_upper = calculate_stats(metrics[metric])
        print(f"{metric}:")
        print(f"  Average: {mean:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

def main():
    # Check if filename is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <json_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        # Read JSON file
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Process each split
        for split in ['train', 'val', 'test']:
            if split in data:
                process_split(data[split], split)
            else:
                print(f"Warning: {split} split not found in the data")
                
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{filename}' is not a valid JSON file")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
