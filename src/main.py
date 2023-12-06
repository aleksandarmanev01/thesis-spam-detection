from sklearn.model_selection import train_test_split

from utils import log
from model_eval import evaluate_lstm_model, evaluate_conventional_models
from data_config import *

RUN_ID = "Run_5"
log("Reading and preprocessing original SMS dataset...")

# Load and preprocess data
data_df = pd.read_csv(DATASET_PATH).dropna()

train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['label'])

for scenario, files in scenarios.items():
    # If the scenario is one of the specified, first evaluate on the original dataset
    if scenario in ['Scenario_OAM_O', 'Scenario_OAS_O', 'Scenario_OGM_O']:
        dataset_name = "original"
        log(f"Processing {dataset_name} for {scenario}...")

        evaluate_conventional_models(train_data.copy(), test_data.copy(), scenario, dataset_name, RUN_ID)
        evaluate_lstm_model(train_data.copy(), test_data.copy(), scenario, dataset_name, RUN_ID)

    # Now process the augmented/generated datasets
    for file_path in files:
        dataset_name = file_path.split('/')[-1].replace('.csv', '')
        log(f"Processing {dataset_name} for {scenario}...")

        generated_by_llm = pd.read_csv(file_path).dropna()
        new_train_data, new_test_data = config_scenario(generated_by_llm, train_data, test_data, scenario)
        evaluate_conventional_models(new_train_data.copy(), new_test_data.copy(), scenario, dataset_name, RUN_ID)
        evaluate_lstm_model(new_train_data.copy(), new_test_data.copy(), scenario, dataset_name, RUN_ID)

log("All scenarios, models and datasets evaluated.")
