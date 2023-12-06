import numpy as np
import torch

from sklearn.metrics import f1_score, accuracy_score


def train_one_epoch(model, optimizer, train_loader, criterion):
    model.train()

    # Lists to hold per-batch metrics
    batch_losses = []
    all_train_preds = []
    all_train_labels = []

    # Loop over each batch from the training set
    for batch_X, batch_y in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)

        # Compute loss
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Save loss for this batch
        batch_losses.append(loss.item())

        # Compute probabilities and predicted labels
        y_pred_prob = torch.sigmoid(outputs)
        y_pred_label = (y_pred_prob > 0.5).float()

        # Collect predicted labels and true labels for calculating F1 at the end of the epoch
        all_train_preds.extend(y_pred_label.detach().numpy())
        all_train_labels.extend(batch_y.detach().numpy())

    # Compute average loss, accuracy and F1 score for this epoch
    epoch_loss = np.mean(batch_losses)

    # Compute accuracy and F1 score for the entire epoch
    epoch_accuracy = accuracy_score(all_train_labels, all_train_preds)
    epoch_f1 = f1_score(all_train_labels, all_train_preds)

    return model, epoch_loss, epoch_accuracy, epoch_f1


def evaluate(model, val_loader, criterion):
    # Validation Loop
    model.eval()

    # Lists to hold per-batch metrics and labels for validation
    val_batch_losses = []
    all_val_preds = []
    all_val_labels = []

    # Loop over each batch from the validation set
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            # Forward pass
            outputs = model(batch_X)

            # Compute loss
            loss = criterion(outputs, batch_y)

            # Save loss for this batch
            val_batch_losses.append(loss.item())

            # Compute probabilities and predicted labels
            y_pred_prob = torch.sigmoid(outputs)
            y_pred_label = (y_pred_prob > 0.5).float()

            # Collect predicted labels and true labels for calculating F1 at the end of the epoch
            all_val_preds.extend(y_pred_label.detach().numpy())
            all_val_labels.extend(batch_y.detach().numpy())

        # Compute average loss for this epoch on the validation set
        val_epoch_loss = np.mean(val_batch_losses)

        return val_epoch_loss, all_val_labels, all_val_preds
