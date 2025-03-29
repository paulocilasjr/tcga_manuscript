import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create metadata CSV (one row per sample with sample_name and label)
metadata = {
    'sample_name': ['sampleA', 'sampleB', 'sampleC', 'sampleD'],
    'label': [0, 1, 0, 1]
}
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv('metadata.csv', index=False)

# Create embeddings CSV (multiple instances per sample with embeddings)
embeddings = {
    'sample_name': [
        'sampleA', 'sampleA', 'sampleA',      # 3 instances for sampleA
        'sampleB', 'sampleB',                 # 2 instances for sampleB
        'sampleC', 'sampleC', 'sampleC', 'sampleC',  # 4 instances for sampleC
        'sampleD', 'sampleD', 'sampleD'       # 3 instances for sampleD
    ],
    'instance_id': [1, 2, 3, 1, 2, 1, 2, 3, 4, 1, 2, 3],
    'feature1': [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
    'feature2': [0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
}
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.to_csv('embeddings.csv', index=False)

print("Test datasets created: 'metadata.csv' and 'embeddings.csv'")
