"""
CortiX Module 3 — Training Loop for PyTorch LSTM-CNN Model

Performs supervised calibration training with cross-entropy loss, class weighting 
to handle imbalance, AdamW optimizer, cosine annealing, and early stopping.
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from cortix.config import config
from cortix.classifier.model import CortixLSTMCNN
from cortix.classifier.dataset import prepare_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.classifier.train")


def train_model(
    csv_path: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    device: str | None = None,
    model_path: str | None = None,
):
    epochs = epochs or config.CLASSIFIER_EPOCHS
    batch_size = batch_size or config.CLASSIFIER_BATCH_SIZE
    lr = lr or config.CLASSIFIER_LR
    weight_decay = weight_decay or config.CLASSIFIER_WEIGHT_DECAY
    
    if model_path is None:
        save_path = config.MODEL_PATH_NSLKDD if "nslkdd" in csv_path.lower() else config.MODEL_PATH_CICIDS2017
    else:
        save_path = model_path
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Using device: %s | Target model checkpoint: %s", device, save_path)

    # 1. Load Data
    train_loader, val_loader, test_loader, encoder, _ = prepare_datasets(
        csv_path,
        seq_len=config.CLASSIFIER_SEQ_LEN,
        batch_size=batch_size,
    )

    num_classes = len(encoder.classes_)
    num_features = getattr(train_loader.dataset, "X", torch.empty((0, config.CLASSIFIER_NUM_FEATURES))).shape[1]
    logger.info("Dataset specs: %d features, %d classes (%s)", num_features, num_classes, encoder.classes_)

    # 2. Build Model
    model = CortixLSTMCNN(num_classes=num_classes, num_features=num_features).to(device)

    # Handle Class Imbalance via Weighted CrossEntropy
    # Count frequencies of each class in training loader
    class_counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for val in labels:
            class_counts[val] += 1
            
    # Class weights inversely proportional to frequencies
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts + 1e-8)
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Early stopping configuration
    best_val_loss = float("inf")
    patience = config.CLASSIFIER_EARLY_STOPPING_PATIENCE
    patience_counter = 0

    # 3. Training Loop
    logger.info("Starting training loop for %d epochs", epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss /= total
        train_acc = 100.0 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                logits = model(data)
                loss = criterion(logits, targets)

                val_loss += loss.item() * data.size(0)
                _, predicted = logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        logger.info(
            "Epoch [%d/%d] | Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%%",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model parameters
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info("Saved best model checkpoint to %s", save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs", epoch)
                break

    logger.info("Training complete. Best model checkpoint at: %s", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cortix LSTM-CNN")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to cleaned dataset CSV"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Target model checkpoint path (.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    args = parser.parse_args()
    train_model(csv_path=args.dataset, model_path=args.model_path, epochs=args.epochs)
