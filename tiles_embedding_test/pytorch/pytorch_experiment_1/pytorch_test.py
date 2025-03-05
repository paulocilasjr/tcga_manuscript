import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
import random

# Custom Dataset Class
class TCGADataset(Dataset):
    def __init__(self, csv_file, split_value):
        print(f"Loading data for split {split_value} from {csv_file}", flush=True)
        df = pd.read_csv(csv_file)
        self.data = df[df['split'] == split_value].reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(f"No data found for split {split_value}")

        print(f"Parsing embeddings for split {split_value}", flush=True)
        self.embeddings = np.array([np.array(x.split(), dtype=np.float32)
                                   for x in self.data['embedding'].values])
        self.labels = self.data['label'].values.astype(np.float32)
        print(f"Normalizing embeddings for split {split_value}", flush=True)
        self.embeddings = (self.embeddings - self.embeddings.mean(axis=0)) / self.embeddings.std(axis=0)
        print(f"Dataset for split {split_value} ready with {len(self.data)} samples", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
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

# Training and Evaluation Functions
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
        optimizer.step()

        running_loss += loss.item() * embeddings.size(0)
        preds = torch.sigmoid(outputs).detach().cpu().numpy() > 0.5
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (i + 1) % 10000 == 0:  # Print every 10,000 batches
            print(f"Batch {i+1}/{len(loader)} completed, running loss: {running_loss / ((i+1) * loader.batch_size):.4f}", flush=True)

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss, accuracy

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
            preds = probs > 0.5
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

            if (i + 1) % 10000 == 0:
                print(f"Evaluation batch {i+1}/{len(loader)} completed", flush=True)

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

# Set random seed
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
    parser = argparse.ArgumentParser(description='Train a TCGA model with PyTorch')
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

    train_dataset = TCGADataset(args.dataset, 0)
    val_dataset = TCGADataset(args.dataset, 1)
    test_dataset = TCGADataset(args.dataset, 2)

    print(f"Train dataset size: {len(train_dataset)}", flush=True)
    print(f"Validation dataset size: {len(val_dataset)}", flush=True)
    print(f"Test dataset size: {len(test_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_dataset.embeddings.shape[1]
    print(f"Input dimension: {input_dim}", flush=True)
    model = TCGAModel(input_dim).to(device)
    print(f"Model initialized and moved to {device}", flush=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience)
    print(f"Optimizer and scheduler initialized with learning rate: {learning_rate}", flush=True)

    best_val_roc_auc = -float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}", flush=True)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}", flush=True)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
              f"ROC-AUC: {val_metrics['roc_auc']:.4f}", flush=True)

        scheduler.step(val_metrics['roc_auc'])
        print(f"Learning rate after epoch {epoch+1}: {scheduler.get_last_lr()[0]}", flush=True)

        if val_metrics['roc_auc'] > best_val_roc_auc:
            best_val_roc_auc = val_metrics['roc_auc']
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model with ROC-AUC:", best_val_roc_auc, flush=True)
        else:
            early_stop_counter += 1
            print(f"Early stop counter: {early_stop_counter}/{early_stop}", flush=True)
            if early_stop_counter >= early_stop:
                print("Early stopping triggered", flush=True)
                break

    print("Training completed. Best Val ROC-AUC:", best_val_roc_auc, flush=True)

    print("Loading best model for final evaluation", flush=True)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    print("Evaluating on train set", flush=True)
    train_metrics = evaluate(model, train_loader, criterion, device)
    print("Evaluating on validation set", flush=True)
    val_metrics = evaluate(model, val_loader, criterion, device)
    print("Evaluating on test set", flush=True)
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\nFinal Metrics:", flush=True)
    for split, metrics in [('Train', train_metrics), ('Validation', val_metrics), ('Test', test_metrics)]:
        print(f"\n{split} Metrics:", flush=True)
        print(f"Loss: {metrics['loss']:.4f}", flush=True)
        print(f"Accuracy: {metrics['accuracy']:.4f}", flush=True)
        print(f"Precision: {metrics['precision']:.4f}", flush=True)
        print(f"Recall: {metrics['recall']:.4f}", flush=True)
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}", flush=True)
        print(f"Specificity: {metrics['specificity']:.4f}", flush=True)
        print(f"F1-Score: {metrics['f1']:.4f}", flush=True)

if __name__ == "__main__":
    main()
