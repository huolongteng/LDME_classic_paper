"""
Step-by-step training script for the BODMAS dataset using a simple DNN.
This script:
1) Loads and samples the dataset.
2) Builds PyTorch datasets and dataloaders.
3) Defines a feed-forward network.
4) Trains with a progress bar.
5) Evaluates the model on the held-out test set.

Note: The script assumes the `metadata/bodmas.npz` file exists and
contains `X` (features) and `y` (labels). No model checkpointing is
performed; only metrics are printed.
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from load_data import BODMASLoader


# -----------------------------
# 1. Configuration
# -----------------------------
# File path configuration
file_path = os.path.join(os.path.dirname(__file__), "..", "metadata", "bodmas.npz")
if not os.path.exists(file_path):
    raise FileNotFoundError(
        "Dataset not found. Expecting metadata/bodmas.npz relative to the repository root."
    )

# Sampling configuration
zero_count = 40000
one_count = 40000
test_size = 0.2

# Training hyperparameters
batch_size = 64
num_epochs = 20
learning_rate = 5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 2. Data Loading and Splitting
# -----------------------------
loader = BODMASLoader(file_path)
# The full dataset is loaded first so that we can draw balanced samples.
_X_full, _y_full = loader.load()
X_sub, y_sub = loader.sample_subset(zero_count=zero_count, one_count=one_count)
X_train, X_test, y_train, y_test = loader.split(X_sub, y_sub, test_size=test_size)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create TensorDataset and DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -----------------------------
# 3. Model Definition
# -----------------------------
class DNN(nn.Module):
    """A simple multi-layer perceptron for binary classification."""

    def __init__(self, input_size: int = 2381, dropout_rate: float = 0.1):
        super().__init__()
        hidden_sizes = [4096, 2048, 1024, 512]
        num_classes = 2

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# -----------------------------
# 4. Training and Evaluation Utilities
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch with a progress bar."""

    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()
        total_samples += inputs.size(0)

        avg_loss = running_loss / total_samples
        avg_acc = running_correct / total_samples
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"})

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, str]:
    """Evaluate the model and return loss, accuracy, and a classification report."""

    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)
    cls_report = classification_report(all_labels, all_preds, digits=4)

    return avg_loss, accuracy, cls_report


# -----------------------------
# 5. Main Training Loop
# -----------------------------
def main() -> None:
    """Main entry point to train and evaluate the DNN model."""

    # Initialize model, loss, and optimizer
    input_size = X_train_tensor.shape[1]
    model = DNN(input_size=input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training epochs with progress bar feedback
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Evaluation on the test set (no checkpointing)
    test_loss, test_acc, test_report = evaluate(model, test_loader, criterion, device)
    print("\n[Evaluation]")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report:\n", test_report)


if __name__ == "__main__":
    main()
