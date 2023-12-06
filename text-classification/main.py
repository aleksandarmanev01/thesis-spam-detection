import pandas as pd
from classification_function import evaluate_model

# Load data
data_df = pd.read_csv('../data/sms_spam_dataset.csv')

# Prompt setup
prompt1 = """You are a security expert specialized in identifying spam SMS messages.

Your task is to analyze the content of the provided SMS message and determine whether it is 'spam' or 'ham'. 

Message: '{message}'
Classification:"""

prompt2 = """You are a security expert specialized in identifying spam SMS messages.

Your task is to analyze the content of the provided SMS message and determine whether it is 'spam' or 'ham'.

Example:

Message: 'Congratulations! You've been selected to receive a $1000 Walmart Gift Card! Claim your prize now!'
Classification: spam

Message: '{message}'
Classification:"""

prompt3 = """You are a security expert specialized in identifying spam SMS messages.

Your task is to analyze the content of the provided SMS message and determine whether it is 'spam' or 'ham'.

Example:

Message: 'Congratulations! You've been selected to receive a $1000 Walmart Gift Card! Claim your prize now!'
Classification: spam

Message: 'Lunch tomorrow at 1 pm? Let's try that new sushi place!'
Classification: ham

Message: '{message}'
Classification:"""

prompt4 = """You are a security expert specialized in identifying spam SMS messages.

Your task is to analyze the content of the provided SMS message and determine whether it is 'spam' or 'ham'.

Example:

Message: 'Lunch tomorrow at 1 pm? Let's try that new sushi place!'
Classification: ham

Message: '{message}'
Classification:"""

# Define models and prompts to evaluate
models = [
    ("google/flan-t5-xxl", prompt1, "./predictions/flan-t5/predictions_prompt1_flan-t5.csv"),
    ("google/flan-t5-xxl", prompt2, "./predictions/flan-t5/predictions_prompt2_flan-t5.csv"),
    ("google/flan-t5-xxl", prompt3, "./predictions/flan-t5/predictions_prompt3_flan-t5.csv"),
    ("google/flan-t5-xxl", prompt4, "./predictions/flan-t5/predictions_prompt4_flan-t5.csv"),
    ("meta-llama/Llama-2-70b-chat-hf", prompt2, "./predictions/llama2/predictions_prompt2_llama2.csv"),
    ("meta-llama/Llama-2-70b-chat-hf", prompt3, "./predictions/llama2/predictions_prompt3_llama2.csv"),
    ("meta-llama/Llama-2-70b-chat-hf", prompt4, "./predictions/llama2/predictions_prompt4_llama2.csv")
]

# Iterate over each model and prompt
for model_name, prompt_template, output_path in models:
    print(f"Starting evaluation with {model_name}:")
    evaluate_model(data_df, prompt_template, model_name, output_path)
    print(f"Completed evaluation with {model_name}.")
