import json
import os
import numpy as np

# Set the directory for results
results_dir = "../../evaluations/"

# Scenarios
scenarios = ['Scenario_OAM_O', 'Scenario_OAS_O', 'Scenario_AM_O',
             'Scenario_OGM_O', 'Scenario_GM_O', 'Scenario_GM_F']

# Initialize a dictionary to store the average results and standard deviation
average_results = {}

# Iterate over each scenario
for scenario in scenarios:
    summed_results = {}
    all_values = {}

    # Read the results of the 5 runs for the current scenario
    for i in range(1, 6):
        with open(os.path.join(results_dir, f"results_{scenario}_run_{i}.json"), "r") as file:
            current_results = json.load(file)

            # For each dataset and model in the current results, sum up the metrics
            for dataset, dataset_results in current_results.items():
                if dataset not in summed_results:
                    summed_results[dataset] = {}
                    all_values[dataset] = {}

                for model, model_results in dataset_results.items():
                    if model not in summed_results[dataset]:
                        summed_results[dataset][model] = {key: 0 for key in model_results.keys()}
                        all_values[dataset][model] = {key: [] for key in model_results.keys()}

                    for metric, value in model_results.items():
                        summed_results[dataset][model][metric] += value
                        all_values[dataset][model][metric].append(value)

    # Compute the average results by dividing by 5
    # Also compute standard deviation for each metric
    average_results[scenario] = {}
    for dataset, dataset_results in summed_results.items():
        average_results[scenario][dataset] = {}
        for model, model_results in dataset_results.items():
            average_results[scenario][dataset][model] = {
                metric: {
                    'average': value / 5,
                    'std_dev': np.std(all_values[dataset][model][metric])
                }
                for metric, value in model_results.items()
            }

# Write the average results with standard deviation to new JSON files
for scenario, results in average_results.items():
    with open(os.path.join(results_dir, f"averages/results_{scenario}_average.json"), "w") as file:
        json.dump(results, file)

print("Average results with standard deviations computed and saved.")
