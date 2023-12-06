import os
import json
import pandas as pd

# Set the directory for results
results_dir = "../../evaluations/averages/"
save_dir = "../../results/Tables/"

# Scenarios
scenarios = ['Scenario_OAM_O', 'Scenario_OAS_O', 'Scenario_AM_O',
             'Scenario_OGM_O', 'Scenario_GM_O', 'Scenario_GM_F']

# Metrics to consider
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

# Iterate through each scenario
for scenario in scenarios:
    with open(os.path.join(results_dir, f"results_{scenario}_average.json"), "r") as file:
        all_results = json.load(file)

    # For each metric, generate a table
    for metric in metrics:
        # Preparing the data
        data = {}
        datasets = list(all_results.keys())
        for dataset in datasets:
            data[dataset] = {}
            for model, model_results in all_results[dataset].items():
                avg_value = model_results[metric]['average']
                std_dev = model_results[metric]['std_dev']
                data[dataset][model] = f"{avg_value*100:.2f}% ± {std_dev*100:.2f}"

        # Convert dictionary to DataFrame
        df = pd.DataFrame(data)

        # Save the table to a file (CSV format is chosen for clarity, but other formats can be used too)
        table_path = os.path.join(save_dir, f"{metric}_{scenario}.csv")
        df.to_csv(table_path)
        print(f"Table for {metric} in {scenario} saved to {table_path}.")

print("All tables generated.")
