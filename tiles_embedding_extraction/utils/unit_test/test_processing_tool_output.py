import os
import sys
import pandas as pd

# Create mock embedding file
embedding_data = {
    'sample_name': ['sample1', 'sample1', 'sample2_suffix1', 'sample2_suffix2', 'sample-3_special', 'no_match_sample'],
    'vector_0': [0.1, 0.4, 0.7, 1.0, 1.3, 1.6],
    'vector_1': [0.2, 0.5, 0.8, 1.1, 1.4, 1.7],
    'vector_2': [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
}
embedding_df = pd.DataFrame(embedding_data)
embedding_df.to_csv('mock_embeddings.csv', index=False)
print("Mock embeddings file created: mock_embeddings.csv", flush=True)

# Create mock metadata file
metadata_data = {
    'sample_name': ['sample1', 'sample2', 'sample-3_special'],
    'label': ['positive', 'negative', 'neutral']
}
metadata_df = pd.DataFrame(metadata_data)
metadata_df.to_csv('mock_metadata.csv', index=False)
print("Mock metadata file created: mock_metadata.csv", flush=True)


