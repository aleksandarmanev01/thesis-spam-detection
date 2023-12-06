import os
import pandas as pd
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def log(message):
    print(f"[LOG] {message}")


def compute_metrics(y_true, y_pred):
    return {
        'F1': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred)
    }


def extract_number_from_name(name):
    """Extracts the number from the name based on different possible formats."""

    if name == "original":
        return -1  # So that "original" always comes first when sorting

    if "generated_" in name and "_each" in name:
        return int(name.split('_')[1])  # For "generated_number_each" pattern

    return int(name.split('_')[-1])  # Default for "*_number" pattern


def process_predictions(directory):
    results = {}

    for scenario in os.listdir(directory):
        scenario_path = os.path.join(directory, scenario)

        if os.path.isdir(scenario_path):
            for dataset in sorted([d for d in os.listdir(scenario_path) if not d.startswith('.')], key=extract_number_from_name):
                dataset_path = os.path.join(scenario_path, dataset)

                if os.path.isdir(dataset_path):
                    for model_file in os.listdir(dataset_path):
                        model_name = model_file.replace('.csv', '')
                        model_path = os.path.join(dataset_path, model_file)

                        df = pd.read_csv(model_path)
                        metrics = compute_metrics(df["actual_label"], df["predicted_label"])
                        results.setdefault(scenario, {}).setdefault(dataset, {})[model_name] = metrics

    return results


def save_results_as_json(results, save_dir, scenario, run_id):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = f"results_{scenario}_run_{run_id}.json"
    path = os.path.join(save_dir, file_name)
    with open(path, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    base_directory = "../../predictions"
    save_directory = "../../evaluations"

    # Loop through all runs
    for i in range(1, 6):  # for Run_1 to Run_5
        current_directory = os.path.join(base_directory, f"Run_{i}")
        log(f"Processing predictions for Run_{i}...")

        # Get results for all scenarios in the current run
        results = process_predictions(current_directory)

        # Save results for each scenario within the run
        for scenario, scenario_results in results.items():
            save_results_as_json(scenario_results, save_directory, scenario, i)

    log("Metrics computation completed.")
