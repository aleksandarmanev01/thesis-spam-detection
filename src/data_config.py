import pandas as pd

DATASET_PATH = '../data/sms_spam_dataset.csv'
AUGMENTED_SPAM_FILES = [f'../data/Augmentation/Spam/augmented_spam_{i}.csv' for i in range(1, 6)]
AUGMENTED_MIXED_FILES = [f'../data/Augmentation/Mixed/augmented_mixed_{i}.csv' for i in
                         [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
GENERATED_FILES = [f'../data/Generation/generated_{i}_each.csv' for i in
                   [350, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]]
LSTM_LOG_FILE_PATH = './lstm/training_data/training_logs/'

scenarios = {
    'Scenario_OAS_O': AUGMENTED_SPAM_FILES,
    'Scenario_OAM_O': AUGMENTED_MIXED_FILES,
    'Scenario_AM_O': ['../data/Augmentation/Mixed/augmented_mixed_100.csv'],
    'Scenario_OGM_O': GENERATED_FILES,
    'Scenario_GM_O': GENERATED_FILES,
    'Scenario_GM_F': GENERATED_FILES
}


def config_scenario(generated_data, original_train, original_test, scenario_type):
    """
       Configure the training and test datasets based on the specified scenario type.

       Parameters:
       - generated_data (DataFrame): The dataframe containing the generated or augmented data.
       - original_train (DataFrame): The dataframe containing the original training data.
       - original_test (DataFrame): The dataframe containing the original test data.
       - scenario_type (str): The type of scenario to configure datasets for.

       Returns:
       - train (DataFrame): Configured training data for the scenario.
       - test (DataFrame): Configured test data for the scenario.
       """
    if scenario_type == "Scenario_OAM_O" or scenario_type == "Scenario_OAS_O" or scenario_type == "Scenario_OGM_O":
        train = pd.concat([original_train, generated_data], axis=0).sample(frac=1).reset_index(drop=True)
        test = original_test
    elif scenario_type == "Scenario_AM_O" or scenario_type == "Scenario_GM_O":
        train = generated_data.sample(frac=1).reset_index(drop=True)
        test = original_test
    elif scenario_type == "Scenario_GM_F":
        train = generated_data.sample(frac=1).reset_index(drop=True)
        test = pd.concat([original_train, original_test], axis=0).sample(frac=1).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return train, test


def preprocess_data(data_set, preprocessing_func):
    """
        Preprocess the dataset and split it into features and labels.

        Parameters:
        - data_set (DataFrame): The data to preprocess.
        - preprocessing_func (function): The function to apply on the 'message' column.

        Returns:
        - X (Series): The processed 'message' column.
        - y (Series): The 'label' column.
    """
    data_set['message'] = data_set['message'].apply(preprocessing_func)
    data_set['label'] = data_set['label'].map({'ham': 0, 'spam': 1})

    X, y = data_set['message'], data_set['label']
    X, y = drop_empty_rows_and_reset_index(X, y)
    return X, y


def drop_empty_rows_and_reset_index(df_X, df_y):
    empty_rows = df_X.index[df_X == '']
    if not empty_rows.empty:
        df_X.drop(empty_rows, inplace=True)
        df_y.drop(empty_rows, inplace=True)
    return df_X.reset_index(drop=True), df_y.reset_index(drop=True)
