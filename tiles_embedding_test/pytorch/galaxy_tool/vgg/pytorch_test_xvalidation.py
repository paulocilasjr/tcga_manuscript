import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import random
from scipy import stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Custom Dataset Class
class TCGADataset(Dataset):
    def __init__(self, df):
        self.data = df

        # Extract features (vector1 to vector25088) and labels
        # Exclude 'sample_name' and 'split' if present, only use vector columns
        vector_cols = [col for col in df.columns if col.startswith('vector')]
        self.embeddings = self.data[vector_cols].values.astype(np.float32)
        self.labels = self.data['label'].values.astype(np.float32)

        # Handle NaNs in embeddings before normalization
        if np.isnan(self.embeddings).sum() > 0:
            logging.warning("NaN values found in embeddings. Replacing with 0.")
            self.embeddings = np.nan_to_num(self.embeddings, nan=0.0)

        # Normalize embeddings, handling zero standard deviation
        means = np.nanmean(self.embeddings, axis=0)  # Use nanmean to ignore any residual NaNs
        stds = np.nanstd(self.embeddings, axis=0)    # Use nanstd for consistency
        stds[stds == 0] = 1  # Prevent division by zero
        self.embeddings = (self.embeddings - means) / stds

        # Final check for NaNs after normalization
        if np.isnan(self.embeddings).sum() > 0:
            logging.error("NaN values still present after normalization.")
            raise ValueError("NaN values in embeddings after normalization.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Model Definition
class TCGAModel(nn.Module):
    def __init__(self, input_dim):
        super(TCGAModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.encoder(x)

# Training Function with Gradient Clipping
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    print(f"Starting training epoch with {len(loader)} batches", flush=True)
    for i, (embeddings, labels) in enumerate(loader):
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * embeddings.size(0)
        preds = torch.sigmoid(outputs).detach().cpu().numpy() > 0.5
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (i + 1) % 10000 == 0:
            print(f"Batch {i+1}/{len(loader)} completed, running loss: {running_loss / ((i+1) * loader.batch_size):.4f}", flush=True)

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss, accuracy

# Evaluation Function with NaN Handling
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    print(f"Starting evaluation with {len(loader)} batches", flush=True)
    with torch.no_grad():
        for i, (embeddings, labels) in enumerate(loader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings).squeeze(-1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * embeddings.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            # Replace NaN probabilities with 0.5 (neutral prediction)
            probs = np.nan_to_num(probs, nan=0.5)
            preds = probs > 0.5
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

            if (i + 1) % 10000 == 0:
                print(f"Evaluation batch {i+1}/{len(loader)} completed", flush=True)

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    # Handle cases where metrics might be undefined
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs) if not np.isnan(all_probs).any() else 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'specificity': specificity,
        'f1': f1
    }

# Set Random Seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main Training Loop
def main():
    parser = argparse.ArgumentParser(description='Train a TCGA model with PyTorch and 10-fold CV')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the CSV dataset file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    print(f"Setting random seed to {args.seed}", flush=True)
    set_seed(args.seed)

    batch_size = 4
    epochs = 50
    learning_rate = 0.0001
    patience = 15
    early_stop = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load CSV
    print(f"Loading CSV from {args.dataset}", flush=True)
    df = pd.read_csv(args.dataset)

    # Prepare for cross-validation
    print("Preparing 10-fold cross-validation", flush=True)
    unique_samples = df.groupby('sample_name').agg({'label': 'first'}).reset_index()
    X = unique_samples['sample_name'].values
    y = unique_samples['label'].values

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold_metrics = {'train': [], 'val': [], 'test': []}

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold}/10 ===", flush=True)

        # Split train_val_idx into train_idx and val_idx
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.1/0.9,  # 10% of 90% train_val for validation
            stratify=y[train_val_idx],
            random_state=args.seed
        )

        # Set 'split' in unique_samples directly
        unique_samples['split'] = -1
        unique_samples.loc[train_idx, 'split'] = 0
        unique_samples.loc[val_idx, 'split'] = 1
        unique_samples.loc[test_idx, 'split'] = 2

        # Merge split back to original DataFrame
        fold_df = df.merge(unique_samples[['sample_name', 'split']], on='sample_name', how='left')

        # Create datasets for this fold
        train_df = fold_df[fold_df['split'] == 0]
        val_df = fold_df[fold_df['split'] == 1]
        test_df = fold_df[fold_df['split'] == 2]

        # Label counts for this fold
        print(f"Fold {fold} - Training label counts: {train_df['label'].value_counts().to_dict()}", flush=True)
        print(f"Fold {fold} - Validation label counts: {val_df['label'].value_counts().to_dict()}", flush=True)
        print(f"Fold {fold} - Test label counts: {test_df['label'].value_counts().to_dict()}", flush=True)

        # Create datasets and loaders
        train_dataset = TCGADataset(train_df)
        val_dataset = TCGADataset(val_df)
        test_dataset = TCGADataset(test_df)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_dim = train_dataset.embeddings.shape[1]
        model = TCGAModel(input_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience)

        best_val_roc_auc = -float('inf')
        early_stop_counter = 0

        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)

            print(f"Fold {fold} - Epoch {epoch+1}/{epochs}", flush=True)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}", flush=True)
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"ROC-AUC: {val_metrics['roc_auc']:.4f}", flush=True)

            scheduler.step(val_metrics['roc_auc'])

            if val_metrics['roc_auc'] > best_val_roc_auc:
                best_val_roc_auc = val_metrics['roc_auc']
                early_stop_counter = 0
                torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop:
                    print(f"Fold {fold} - Early stopping triggered", flush=True)
                    break

        # Evaluate all splits with best model
        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth', weights_only=True))
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        fold_metrics['train'].append(train_metrics)
        fold_metrics['val'].append(val_metrics)
        fold_metrics['test'].append(test_metrics)

        # Print detailed metrics for all splits
        print(f"\nFold {fold} - Train Metrics:", flush=True)
        for key, value in train_metrics.items():
            print(f"  {key.capitalize()}: {value:.4f}", flush=True)

        print(f"\nFold {fold} - Validation Metrics:", flush=True)
        for key, value in val_metrics.items():
            print(f"  {key.capitalize()}: {value:.4f}", flush=True)

        print(f"\nFold {fold} - Test Metrics:", flush=True)
        for key, value in test_metrics.items():
            print(f"  {key.capitalize()}: {value:.4f}", flush=True)

    # Compute and print average metrics and 95% CI for all splits at the end
    metrics_keys = ['loss', 'accuracy', 'precision', 'recall', 'roc_auc', 'specificity', 'f1']
    split_names = ['Train', 'Validation', 'Test']

    print("\n=== Final 10-Fold Cross-Validation Summary ===", flush=True)
    for split_idx, split in enumerate(split_names):
        split_key = split.lower()
        # Collect all values for each metric across folds
        metric_values = {key: [m[key] for m in fold_metrics[split_key]] for key in metrics_keys}
        
        print(f"\n10-Fold Cross-Validation Results ({split} Set):")
        print("Average Metrics:")
        for key in metrics_keys:
            mean_val = np.mean(metric_values[key])
            # Using percentile method for 95% CI (non-parametric, no assumption of normality)
            ci_lower = np.percentile(metric_values[key], 2.5)
            ci_upper = np.percentile(metric_values[key], 97.5)
            print(f"  {key.capitalize()}: {mean_val:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")

if __name__ == "__main__":
    main()
