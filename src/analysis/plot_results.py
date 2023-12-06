import os
import json
import matplotlib.pyplot as plt

# Set the directory for results
results_dir = "../../evaluations/averages/"
save_dir = "../../results/Plots/"

# Scenarios
scenarios = ['Scenario_OAM_O', 'Scenario_OAS_O', 'Scenario_AM_O',
             'Scenario_OGM_O', 'Scenario_GM_O', 'Scenario_GM_F']

# Metrics to consider (assuming standard names, you might need to update this list based on your actual metrics)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']  # Add or remove metrics as needed

# Iterate through each scenario
for scenario in scenarios:
    with open(os.path.join(results_dir, f"results_{scenario}_average.json"), "r") as file:
        all_results = json.load(file)

    # For each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))

        # Get dataset names directly from the loaded results
        datasets = list(all_results.keys())

        # Iterate through each model
        # Assuming all datasets and models are present in the results, extract model names from the first dataset
        for model in all_results[datasets[0]].keys():
            # Extract the 'average' value for each metric
            averages = [all_results[dataset][model][metric]['average'] for dataset in datasets]
            # Extract standard deviations for each metric
            std_devs = [all_results[dataset][model][metric]['std_dev'] for dataset in datasets]

            plt.plot(datasets, averages, marker='o', label=model)
            plt.fill_between(datasets, [avg - std for avg, std in zip(averages, std_devs)],
                             [avg + std for avg, std in zip(averages, std_devs)], alpha=0.2)

        plt.title(f"Results for {metric} in {scenario}")
        plt.xlabel("Dataset")
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)  # Rotate x-tick labels by 45 degrees
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_{scenario}.png"))
        plt.show()
