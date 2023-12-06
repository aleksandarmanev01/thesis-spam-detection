from conventional_models.models_config import *
from conventional_models.train_eval_conventional import train_and_predict_conventional
from conventional_models.preprocessing_conventional import preprocess_text_baseline
from lstm.dataset_setup import setup_loaders
from lstm.preprocessing_lstm import preprocess_text_lstm
from lstm.model_definition import SpamTextClassifier
from lstm.model_config import *
from lstm.training_evaluation_lstm import *
from utils import log, save_predictions_and_labels, log_header, log_training_data, log_evaluation_results, plot_training_metrics
from data_config import preprocess_data


def evaluate_conventional_models(train_set, test_set, scenario, dataset_name, run_id):
    """
        Evaluate conventional ML models on the provided datasets.

        Parameters:
        - train_set (DataFrame): Training data.
        - test_set (DataFrame): Test data.
        - scenario (str): Current evaluation scenario.
        - dataset_name (str): Name of the dataset being used.
        - run_id (str): Identifier for the current run.
    """
    X_train, y_train = preprocess_data(train_set, preprocess_text_baseline)
    X_test, y_test = preprocess_data(test_set, preprocess_text_baseline)

    for model_name, model in models.items():
        y_pred = train_and_predict_conventional(model, X_train, y_train, X_test, model_name)
        save_predictions_and_labels(X_test, y_pred, y_test, model_name, scenario, dataset_name, run_id)
        log(f"Saved predictions for {model_name} under {scenario}, dataset: {dataset_name}")


def evaluate_lstm_model(train_set, val_set, scenario, dataset_name, run_id):
    """
        Evaluate the LSTM model on the provided datasets.

        Parameters:
        - train_set (DataFrame): Training data.
        - val_set (DataFrame): Validation data.
        - scenario (str): Current evaluation scenario.
        - dataset_name (str): Name of the dataset being used.
        - run_id (str): Identifier for the current run.
    """
    log_header("LSTM", scenario, dataset_name, run_id)
    log(f"Evaluating LSTM under {scenario}, dataset: {dataset_name}")

    X_train, y_train = preprocess_data(train_set, preprocess_text_lstm)
    X_val, y_val = preprocess_data(val_set, preprocess_text_lstm)

    vocab, train_loader, val_loader = setup_loaders(X_train, y_train, X_val, y_val)

    # Initialize model, criterion, optimizer
    model = SpamTextClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, DROPOUT)
    criterion = get_criterion(y_train)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # Training loop
    train_epoch_losses, train_epoch_accuracies, train_epoch_f1_scores = [], [], []
    val_epoch_losses, val_epoch_accuracies, val_epoch_f1_scores = [], [], []

    best_val_loss = float('inf')  # Initialize with positive infinity since we want to minimize loss
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model, epoch_loss, epoch_accuracy, epoch_f1 = train_one_epoch(model, optimizer, train_loader, criterion)

        train_epoch_losses.append(epoch_loss)
        train_epoch_accuracies.append(epoch_accuracy)
        train_epoch_f1_scores.append(epoch_f1)

        log(f"Epoch {epoch + 1}, Train Loss: {epoch_loss}, Train Accuracy: {epoch_accuracy}, Train F1 Score: {epoch_f1}")

        val_epoch_loss, all_val_labels, all_val_preds = evaluate(model, val_loader, criterion)

        val_epoch_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_epoch_f1_score = f1_score(all_val_labels, all_val_preds)

        val_epoch_losses.append(val_epoch_loss)
        val_epoch_accuracies.append(val_epoch_accuracy)
        val_epoch_f1_scores.append(val_epoch_f1_score)

        log(f"Epoch {epoch + 1}, Val Loss: {val_epoch_loss}, Val Accuracy: {val_epoch_accuracy}, Val F1 Score: {val_epoch_f1_score}")
        log('-' * 60)

        # Log training data to the log file
        log_training_data(epoch, epoch_loss, epoch_accuracy, epoch_f1, val_epoch_loss, val_epoch_accuracy, val_epoch_f1_score, run_id)

        scheduler.step(val_epoch_loss)

        # Early stopping logic based on validation loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            # Save the model when validation loss improves
            torch.save(model.state_dict(), './lstm/best_lstm_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE:
                print("Early stopping!")
                break

    # Load the best model
    model.load_state_dict(torch.load('./lstm/best_lstm_model.pth'))
    model.eval()

    # Evaluate on validation set
    _, all_val_labels, all_val_preds = evaluate(model, val_loader, criterion)

    int_val_preds = np.concatenate(all_val_preds).astype(int)
    int_val_labels = np.concatenate(all_val_labels).astype(int)

    save_predictions_and_labels(X_val, int_val_preds, int_val_labels, "LSTM", scenario, dataset_name, run_id)
    log(f"Saved predictions for LSTM under {scenario}, dataset: {dataset_name}")

    # Log the evaluation results to the log file
    log_evaluation_results(all_val_labels, all_val_preds, "LSTM", scenario, dataset_name, run_id)

    # Plot the training and validation process
    plot_training_metrics(train_epoch_losses, val_epoch_losses,
                          train_epoch_accuracies, val_epoch_accuracies,
                          train_epoch_f1_scores, val_epoch_f1_scores,
                          scenario, dataset_name, run_id)
