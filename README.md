# thesis-spam-detection
This repository contains the code and results from my work on the "Spam detection" use case, a component of my Bachelor's thesis on topic "Applications of Large Language Models (LLMs) in Cybersecurity".

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Text Classification](#text-classification)
4. [Data Preparation](#data-preparation)
5. [Scenarios](#scenarios)
6. [Model Evaluation](#model-evaluation)


## Introduction

This repository allows you to reproduce the results from the "Spam Detection" use case explored in my Bachelor's Thesis. The focus of this project is on leveraging Large Language Models (LLMs) for various tasks related to spam detection in the context of cybersecurity.

In particular, you can perform the following tasks:
- Use LLMs for **Text Classification**: Classify messages from the __UCI SMS Spam Collection__ as *spam* or *ham* with state-of-the-art LLMs without fine-tuning.
- Use LLMs to **augment** existing dataset with different strategies as well as **generate** synthetic data.
- Evaluate the **quality** of the LLM-produced data on a set of supervised ML models, based on different scenarios.

## Environment Setup
<details>
<summary>Click to expand</summary>

To get this project up and running, you'll need to set up a virtual environment with all the required dependencies. We use Conda for managing our environment.

1. **Clone the repository**:

First, clone the project repository to your local machine.

```
git clone https://github.com/aleksandarmanev01/thesis-spam-detection.git
cd thesis-spam-detection
```

2. **Create the Conda Environment**:

Use the `environment.yml` file to create a new Conda environment with all required dependencies.

```
conda env create -f environment.yml
```

3. **Activate the Environment**:

After creating the environment, activate it to use.
```
conda activate my-env-name
```
Replace `my-env-name` with the name of your environment.

4. **Verify the Environment** (Optional):

To ensure the environment is set up correctly, you can list all available environments.
```
conda env list
```

5. **Deactivate the Environment**:

When you're done working, deactivate the environment.

```
conda deactivate
```
</details>

## Text Classification
<details>
<summary>Click to expand</summary>

As part of my thesis, two LLMs were utilized for the text classification task: Llama 2 70B Chat and FLAN-T5 XXL.
In particular, we utilized their versions hosted by *Hugging Face*, thus for making inference with Llama 2 70B Chat, a **Pro** subscription is required.  
For each data point of the **UCI SMS Spam Collection**, we made a prediction with the two models using various prompting strategies.
In order to reproduce the work, please do the following: 

1. Create a `config.py` file in `thesis-spam-detection`.
2. Put a set access tokens for *Hugging Face* in the format ``
API_TOKENS= [...]
``  
We evaluate the data points in batches of 1000 to avoid rate limits by *Hugging Face*. Therefore, at least 6 tokens are recommended in order to evaluate all data points in one run.  
**Important:** Due to the sensitivity of the content of `config.py`, it is not tracked by Git.
3. Run `text-classification/main.py`
4. For each data point, a prediction will be made with the two models and the different prompting strategies. The predictions are then saved in `text-classification/predictions/` in the format `message`, `predicted_label`, `actual_label`.
5. Analyse the predictions using the Jupyter Notebooks in the `analysis` folder.
</details>

## Data Preparation
<details>
<summary>Click to expand</summary>  

In order to augment the existing training data with different scenarios and to generate synthetic data, we utilized Llama 2 70B Chat.
The relevant scripts are in the `data` folder.
</details>

## Scenarios
<details>
<summary>Click to expand</summary>

To evaluate the generated data, we followed a systematic approach: adding different portions of augmented or synthetic data to the original training dataset.

As training set, we utilized a fixed 80% of the UCI dataset. In the following, we will refer to it as **TD**.  
As validation set, we utilized the remaining 20% of the UCI dataset. In the following, we will refer to it as **VD**.

The scenarios that are evaluated in this use case are:

| Scenario Name  | Training data          | Validation data |
|----------------|------------------------|-----------------|
| Scenario_OAM_O | TD + Augmented Mixed   | VD              |
| Scenario_OAS_O | TD + Augmented Spam    | VD              |
| Scenario_AM_O  | Augmented Mixed (100%) | VD              |
| Scenario_OGM_O | TD + Generated Mixed   | VD              |
| Scenario_GM_O  | Generated Mixed        | VD              | 
| Scenario_GM_F  | Generated Mixed        | TD + VD         | 

*Note:* **Mixed** refers to augmenting/generating both classes equally.  
The name of the scenario can be understood the following way:
`Scenario_{training data}_{validation data}`, with 
- **OAM**: Original + Augmented Mixed
- **OAS**: Original + Augmented Spam
- **OGM**: Original + Generated Mixed
- **AM**: Augmented Mixed
- **GM**: Generated Mixed
- **O**: Original
</details>

## Model evaluation
<details>
<summary>Click to expand</summary>

In order to evaluate the quality of the LLM-produced data, we utilize a wide range of supervised ML models: BERT, LSTM, NB, LR, KNN, SVM, XGBoost and LightGBM. To reproduce the results in the thesis, please do the following:

First, due to the need for GPU for more efficient evaluation of the BERT model, it is done externally via Google Colab.
Hence, we evaluate BERT first:

1. Open `src/bert/BERT_TextClassifier_SD.ipynb`
2. Connect Google Drive to Google Colab.
3. In My Drive on Google Drive, create the following folders:
`MyDrive/Bachelor's Thesis/Spam detection`
4. In the `Spam detection` folder, please create the following structure:
- `data`: a folder, which should contain the exact same content as the `data` folder in this repo
- `bert`: a folder, which should contain:
  - `training_data`: a folder, which will contain:
    - all log files containing training details for each scenario and data portions across the different runs 
    - `training_plots`, a folder which will contain plots illustrating the training process using different metrics, such as accuracy, loss and F1 score
- `predictions`: a folder, in which all the predictions will be saved automatically
5. Having done this, execute the Jupyter Notebook for the desired amount of runs. Do not forget to adjust the `run_id` properly, the desired format is `Run_{id}`. All scenarios will be evaluated with all portions of data. The training process can be observed on Colab.
6. In the `training_data` on Drive, all the relevant training data will be saved for further analysis (in this repo, this is the data saved in `src/bert/training_data`).
7. In `predictions`, all the predictions made will be saved.

Once the Colab finishes execution for the desired amount of runs, please do the following. Download the `predictions` folder and place it in this repo.
Do not modify the structure, it will be used here as well.
Next, follow the steps:

1. Open `src/main.py` and adjust the `run_id`. The format, once again, is `Run_{id}`.
2. Run `src/main.py`. This will cause all models to be evaluated for all scenarios with all amounts of data.
3. Again, during training, the current state will be logged on the console. Similarly, training data for LSTM will be saved as logs and plots (different metrics) in `lstm/training_data`.
4. While the program is running, predictions will be saved in the `predictions` folder. They have the format: `message`, `predicted_label`, `actual_label`, same as for BERT and **Text Classification**.
5. The predictions are sorted by run id, scenario, dataset, model. 
6. Once the desired amount of runs is done (same as for BERT), navigate to `src/analysis:`
7. Execute `process_predictions.py`, which will compute the metrics that are defined there based on the predictions for each run and scenario.
8. Then, to calculate the average across the different runs, execute `compute_average.py`. This will result in different metrics per scenario.
9. Then, run `generate_tables.py` to generate tables with all the metrics with average results per scenario.
10. Alternatively, run `plot_results.py` to generate plots of the metrics.
11. The tables can be found in `/results/Tables/` and the plots in `/results/Plots/`.
</details>