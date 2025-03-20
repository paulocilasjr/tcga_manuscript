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

# Custom Dataset Class with on-the-fly normalization
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
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)

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

# Main Function with 10-Fold Cross-Validation (Data Leakage Allowed)
def main():
    parser = argparse.ArgumentParser(description='Train a model with 10-fold CV (data leakage allowed)')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the CSV dataset file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    print(f"Setting random seed to {args.seed}", flush=True)
    set_seed(args.seed)

    # Configuration
    batch_size = 1024
    epochs = 50
    learning_rate = 0.0001
    patience = 15
    early_stop = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load CSV
    print(f"Loading CSV from {args.dataset}", flush=True)
    df = pd.read_csv(args.dataset)
    print(f"Data loaded: {df.shape[0]} instances", flush=True)

    # Extract Features and Labels
    feature_cols = [col for col in df.columns if col.startswith("vector")]
    features = df[feature_cols].values.astype(np.float32)
    labels = df['bag_label'].values.astype(np.float32)

    # Prepare for 10-Fold Cross-Validation
    indices = np.arange(len(df))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold_metrics = {'train': [], 'val': [], 'test': []}

    print(f"Starting 10-fold cross-validation", flush=True)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels), 1):
        print(f"\nStarting Fold {fold}/10", flush=True)

        # Split train_val_idx into train_idx and val_idx
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.1/0.9,  # ~11.11% of 90% for validation
            stratify=labels[train_val_idx],
            random_state=args.seed
        )

        # Assign indices
        train_indices = train_idx
        val_indices = val_idx
        test_indices = test_idx

        print(f"Fold {fold}: Train instances: {len(train_indices)}, Val instances: {len(val_indices)}, Test instances: {len(test_indices)}", flush=True)

        # Normalize Using Training Set Statistics
        train_features = features[train_indices]
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0)

        # Create Datasets
        train_dataset = TCGADataset(features, labels, train_indices, mean, std)
        val_dataset = TCGADataset(features, labels, val_indices, mean, std)
        test_dataset = TCGADataset(features, labels, test_indices, mean, std)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize Model
        input_dim = features.shape[1]
        model = TCGAModel(input_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience)

        best_val_roc_auc = -float('inf')
        early_stop_counter = 0

        print(f"Starting training for Fold {fold}", flush=True)
        # Training Loop
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

        print(f"Finished training for Fold {fold}", flush=True)
        # Evaluate All Splits with Best Model
        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth', weights_only=True))
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        fold_metrics['train'].append(train_metrics)
        fold_metrics['val'].append(val_metrics)
        fold_metrics['test'].append(test_metrics)

        print(f"Fold {fold} - Test Metrics:", flush=True)
        for key, value in test_metrics.items():
            print(f"{key.capitalize()}: {value:.4f}", flush=True)

    # Compute and Display Average Metrics with 95% Confidence Intervals
    metrics_keys = ['loss', 'accuracy', 'precision', 'recall', 'roc_auc', 'specificity', 'f1']
    split_names = ['Train', 'Val', 'Test']

    for split in split_names:
        split_key = split.lower()
        avg_metrics = {key: np.mean([m[key] for m in fold_metrics[split_key]]) for key in metrics_keys}
        std_metrics = {key: np.std([m[key] for m in fold_metrics[split_key]]) for key in metrics_keys}

        # 95% CI using t-distribution (10 folds, df=9)
        confidence_level = 0.95
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=9)
        ci_metrics = {}
        for key in metrics_keys:
            margin_error = t_critical * std_metrics[key] / np.sqrt(10)
            ci_metrics[key] = (avg_metrics[key] - margin_error, avg_metrics[key] + margin_error)

        print(f"\n10-Fold Cross-Validation Results ({split} Set):", flush=True)
        print("Average Metrics with 95% CI:", flush=True)
        for key in metrics_keys:
            print(f"{key.capitalize()}: {avg_metrics[key]:.4f} ({ci_metrics[key][0]:.4f}, {ci_metrics[key][1]:.4f})", flush=True)

if __name__ == "__main__":
    main()
