import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
import random

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

# Main Function
def main():
    parser = argparse.ArgumentParser(description='Train a model using predefined splits')
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

    # Load Dataset
    print(f"Loading CSV from {args.dataset}", flush=True)
    df = pd.read_csv(args.dataset)
    print(f"Data loaded: {df.shape[0]} instances", flush=True)

    # Extract Features, Labels, and Splits
    feature_cols = [col for col in df.columns if col.startswith("vector")]
    features = df[feature_cols].values.astype(np.float32)
    labels = df['bag_label'].values.astype(np.float32)
    splits = df['split'].values

    # Assign Indices Based on Predefined Splits
    train_indices = np.where(splits == 0)[0]
    val_indices = np.where(splits == 1)[0]
    test_indices = np.where(splits == 2)[0]

    print(f"Train instances: {len(train_indices)}, Val instances: {len(val_indices)}, Test instances: {len(test_indices)}", flush=True)

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

    print(f"Starting training", flush=True)
    # Training Loop
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}", flush=True)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}", flush=True)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
              f"ROC-AUC: {val_metrics['roc_auc']:.4f}", flush=True)

        scheduler.step(val_metrics['roc_auc'])

        if val_metrics['roc_auc'] > best_val_roc_auc:
            best_val_roc_auc = val_metrics['roc_auc']
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop:
                print(f"Early stopping triggered", flush=True)
                break

    print(f"Finished training", flush=True)
    # Evaluate All Splits with Best Model
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    # Print Metrics
    for split, metrics in zip(['Train', 'Validation', 'Test'], [train_metrics, val_metrics, test_metrics]):
        print(f"\n{split} Set Metrics:", flush=True)
        for key, value in metrics.items():
            print(f"{key.capitalize()}: {value:.4f}", flush=True)

if __name__ == "__main__":
    main()
