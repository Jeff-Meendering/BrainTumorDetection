import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import os
import time
from datetime import datetime
from .model import get_model

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    save_path: str = "models/best_model.pth"
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scaler = amp.GradScaler(enabled=(device.type != 'cpu'))

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    best_val_accuracy = 0.0

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            with amp.autocast(enabled=(device.type != 'cpu')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1) 
                with amp.autocast(enabled=(device.type != 'cpu')):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)

                preds = torch.sigmoid(outputs) > 0.5
                val_total += labels.size(0)
                val_correct += (preds.float() == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_accuracy = val_correct / val_total if val_total > 0 else 0
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Duration: {epoch_duration:.2f}s | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_accuracy:.4f}")

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1}: New best model saved to {save_path} (Val Acc: {best_val_accuracy:.4f})")

        scheduler.step()

    print(f"\nTraining finished. Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Best model weights saved to: {save_path}")

    return history

if __name__ == '__main__':
    print("Trainer script executed directly. This section is for testing/example.")