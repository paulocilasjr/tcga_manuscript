import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
import random
from scipy import stats

# Custom Dataset Class
class TCGADataset(Dataset):
    def __init__(self, features, labels, indices, mean, std):
        self.features = features
        self.labels = labels
        self.indices = indices
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        feature_idx = self.indices[idx]
        features = (self.features[feature_idx] - self.mean) / self.std
        embedding = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(self.labels[feature_idx], dtype=torch.float32)
        return embedding, label

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

# Training Function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for embeddings, labels in loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * embeddings.size(0)
        preds = torch.sigmoid(outputs).detach().cpu().numpy() > 0.5
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss, accuracy

# Evaluation Function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings).squeeze(-1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * embeddings.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = probs > 0.5
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
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

# Set Random Seed for Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_single_seed(seed, args, device):
    print(f"\nStarting run with seed {seed}", flush=True)
    set_seed(seed)

    # Load Dataset
    df = pd.read_csv(args.dataset)
    feature_cols = [col for col in df.columns if col.startswith("vector")]
    features = df[feature_cols].values.astype(np.float32)
    labels = df['bag_label'].values.astype(np.float32)
    splits = df['split'].values

    # Assign Indices
    train_indices = np.where(splits == 0)[0]
    val_indices = np.where(splits == 1)[0]
    test_indices = np.where(splits == 2)[0]

    # Normalize
    train_features = features[train_indices]
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)

    # Create Datasets and Loaders
    batch_size = 1024
    train_dataset = TCGADataset(features, labels, train_indices, mean, std)
    val_dataset = TCGADataset(features, labels, val_indices, mean, std)
    test_dataset = TCGADataset(features, labels, test_indices, mean, std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    input_dim = features.shape[1]
    model = TCGAModel(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    # Training Loop
    best_val_roc_auc = -float('inf')
    early_stop_counter = 0
    epochs = 50

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['roc_auc'])

        if val_metrics['roc_auc'] > best_val_roc_auc:
            best_val_roc_auc = val_metrics['roc_auc']
            early_stop_counter = 0
            torch.save(model.state_dict(), f'best_model_seed_{seed}.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= 30:
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load(f'best_model_seed_{seed}.pth', weights_only=True))
    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    return train_metrics, val_metrics, test_metrics

def main():
    parser = argparse.ArgumentParser(description='Train a model with multiple random seeds')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the CSV dataset file')
    parser.add_argument('--n_seeds', type=int, default=10, help='Number of random seeds to use')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Generate truly random seeds
    random.seed()  # Initialize with system time
    seeds = [random.randint(0, 1000000) for _ in range(args.n_seeds)]
    print(f"Using random seeds: {seeds}", flush=True)

    all_metrics = {'train': [], 'val': [], 'test': []}

    # Run for each seed
    for seed in seeds:
        train_metrics, val_metrics, test_metrics = run_single_seed(seed, args, device)
        all_metrics['train'].append(train_metrics)
        all_metrics['val'].append(val_metrics)
        all_metrics['test'].append(test_metrics)

    # Compute averages and 95% CI
    def compute_stats(metrics_list):
        keys = metrics_list[0].keys()
        means = {}
        cis = {}
        for key in keys:
            values = np.array([m[key] for m in metrics_list])
            mean = np.mean(values)
            
            # Check for edge cases
            if len(values) < 2:  # Need at least 2 samples for CI
                means[key] = mean
                cis[key] = (np.nan, np.nan)
                print(f"Warning: Insufficient samples ({len(values)}) for {key} CI computation.")
                continue
            if np.isnan(mean):  # Skip if mean is NaN
                means[key] = mean
                cis[key] = (np.nan, np.nan)
                print(f"Warning: NaN mean for {key}.")
                continue
            if np.all(values == values[0]):  # All values identical
                means[key] = mean
                cis[key] = (mean, mean)
                print(f"Warning: All values identical for {key}.")
                continue
                
            # Calculate 95% CI using t-distribution
            try:
                ci = stats.t.interval(
                    confidence=0.95,
                    df=len(values)-1,
                    loc=mean,
                    scale=stats.sem(values)
                )
                means[key] = mean
                cis[key] = (ci[0], ci[1])
            except ValueError as e:
                means[key] = mean
                cis[key] = (np.nan, np.nan)
                print(f"Warning: CI computation failed for {key}: {e}")
                
        return means, cis

    # Print results
    for split in ['train', 'val', 'test']:
        means, cis = compute_stats(all_metrics[split])
        print(f"\nAverage {split.capitalize()} Set Metrics over {args.n_seeds} seeds:", flush=True)
        for key in means.keys():
            print(f"{key.capitalize()}: {means[key]:.4f} (95% CI: {cis[key][0]:.4f} - {cis[key][1]:.4f})", flush=True)

if __name__ == "__main__":
    main()
