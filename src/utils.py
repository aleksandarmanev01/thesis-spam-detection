import os
import pandas as pd
import numpy as np
from data_config import LSTM_LOG_FILE_PATH
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

SEPARATOR = '=' * 80
SUB_SEPARATOR = '-' * 80


def save_predictions_and_labels(messages, predictions, labels, model_name, scenario, dataset_name, run_id):
    """
    Save model predictions and true labels to a CSV file.
    """
    directory = f"../predictions/{run_id}/{scenario}/{dataset_name}"

    # Create directories if they don't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = f"{directory}/{model_name}.csv"
    df = pd.DataFrame({
        'message': messages,
        'predicted_label': predictions,
        'actual_label': labels
    })
    df.to_csv(path, index=False)


def log(message):
    print(f"[LOG] {message}")


def log_to_file(content, run_id, directory=LSTM_LOG_FILE_PATH):
    """Helper function to log content to a file."""
    log_file_path = os.path.join(directory, f"training_log_{run_id}.txt")

    # Check if the file exists, if not create it with write mode, else append to it
    mode = 'w' if not os.path.exists(log_file_path) else 'a'

    with open(log_file_path, mode) as log_file:
        log_file.write(content)


def log_header(model_name, scenario, dataset_name, run_id):
    """Logs the header section for each model-scenario-dataset combination."""
    header_content = (
        f"{SEPARATOR}\n"
        f"Model Name: {model_name}\n"
        f"Scenario: {scenario}\n"
        f"Dataset: {dataset_name}\n"
        f"{SEPARATOR}\n"
    )
    log_to_file(header_content, run_id)


def log_training_data(epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, run_id):
    """Logs training and validation data for a specific epoch."""
    training_data_content = (
        f"Epoch {epoch + 1}, Train Loss: {train_loss:.8f}, Train Accuracy: {train_accuracy:.8f}, Train F1 Score: {train_f1:.8f}\n"
        f"Epoch {epoch + 1}, Val Loss: {val_loss:.8f}, Val Accuracy: {val_accuracy:.8f}, Val F1 Score: {val_f1:.8f}\n"
        f"{SUB_SEPARATOR}\n"
    )

    log_to_file(training_data_content, run_id)


def log_evaluation_results(all_val_labels, all_val_preds, model_name, scenario, dataset_name, run_id):
    """Logs the final results using classification report."""
    report = classification_report(all_val_labels, all_val_preds)
    evaluation_content = (
        f"Training for {model_name} under scenario: {scenario}, dataset: {dataset_name} completed.\n"
        f"Evaluation Results:\n"
        f"{SUB_SEPARATOR}\n"
        f"{report}\n"
    )
    log_to_file(evaluation_content, run_id)


def plot_training_metrics(train_epoch_losses, val_epoch_losses,
                          train_epoch_accuracies, val_epoch_accuracies,
                          train_epoch_f1_scores, val_epoch_f1_scores,
                          scenario, dataset_name, run_id):
    """
    Plot the training and validation metrics including loss, accuracy, and F1 score.
    """

    SAVE_DIR = "./lstm/training_data/training_plots/"

    # If save directory doesn't exist, create it
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    epochs = np.arange(len(train_epoch_losses))

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_epoch_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_epoch_losses, label='Validation Loss', marker='o')
    plt.title(f'Loss over Epochs for LSTM with {scenario} on {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    loss_save_path = os.path.join(SAVE_DIR, f'loss_plot_{scenario}_{dataset_name}_{run_id}.png')
    plt.savefig(loss_save_path)
    plt.close()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_epoch_accuracies, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_epoch_accuracies, label='Validation Accuracy', marker='o')
    plt.title(f'Accuracy over Epochs for LSTM with {scenario} on {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    accuracy_save_path = os.path.join(SAVE_DIR, f'accuracy_plot_{scenario}_{dataset_name}_{run_id}.png')
    plt.savefig(accuracy_save_path)
    plt.close()

    # Plot training and validation F1 score
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_epoch_f1_scores, label='Training F1 Score', marker='o')
    plt.plot(epochs, val_epoch_f1_scores, label='Validation F1 Score', marker='o')
    plt.title(f'F1 Score over Epochs for LSTM with {scenario} on {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.xticks(epochs)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    f1_save_path = os.path.join(SAVE_DIR, f'f1_plot_{scenario}_{dataset_name}_{run_id}.png')
    plt.savefig(f1_save_path)
    plt.close()
