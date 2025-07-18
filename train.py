import os
import time

import torch

from utils import get_model, MODEL_PATH, NUM_EPOCHS, LOSS_LOG_PATH, BATCH_SIZE, BEST_MODEL_PATH


def train():
    model, device, criterion, optimizer, train_loader, val_loader, train_dataset = get_model()

    # Resume from checkpoint if exists
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint.get('batch', 0)
            print(f"Resumed from epoch {start_epoch}, batch {start_batch}")
        except (KeyError, RuntimeError, FileNotFoundError, torch.serialization.pickle.UnpicklingError) as e:
            print(f"Failed to load checkpoint from {MODEL_PATH}: {e}")
            start_epoch = 0
            start_batch = 0
    else:
        start_epoch = 0
        start_batch = 0

    # Training loop
    best_loss = float('inf')
    patience = 3
    no_improve_epochs = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Initialize loss log
        if not os.path.exists(LOSS_LOG_PATH):
            with open(LOSS_LOG_PATH, "w") as f:
                f.write("epoch,train_loss,time_sec,timestamp\n")

        model.train()
        running_loss = 0.0
        start_time = time.time()

        for i, (images, labels) in enumerate(train_loader):
            if epoch == start_epoch and i < start_batch:
                continue

            step_start = time.time()

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            step_time = time.time() - step_start
            images_processed = (i + 1) * BATCH_SIZE
            print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}, Time per step: {step_time:.2f}s, "
                f"Images processed: {images_processed}")

            # Save checkpoint after each batch
            checkpoint = {
                'epoch': epoch,
                'batch': i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }
            torch.save(checkpoint, MODEL_PATH)

        epoch_loss = running_loss / len(train_dataset)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f}s")

        # Log epoch loss
        with open(LOSS_LOG_PATH, "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{epoch+1},{epoch_loss:.6f},{elapsed_time:.2f},{timestamp}\n")

        print(f"Logged loss for epoch {epoch+1} to {LOSS_LOG_PATH}")

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve_epochs = 0
            # Save best model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, BEST_MODEL_PATH)
            print(f"Best model saved at epoch {epoch + 1} with loss {epoch_loss:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s)")

            if no_improve_epochs >= patience:
                print("Early stopping: training stopped due to no improvement.")
                break
        # Reset batch count after epoch
        start_batch = 0

if __name__ == "__main__":
    train()
