import time
import re
import pandas as pd

import config
from huggingface_hub import InferenceClient


# Function to generate a consistent seed value based on input k, variant_num allows for different seeds for the same k
def generate_seed(k, variant_num):
    base_seed = 42
    return base_seed * variant_num + k


# Function to extract a message from a text string using regex, capturing text within quotes
def extract_augmented_message(text):
    match = re.search(r"\"(.*?)\"", text)
    if match:
        return match.group(1)
    else:
        return None


# Function to augment a message using a large language model, with retry logic
def augment_with_LLM(message, seed, attempt=0, max_attempts=5):
    if attempt >= max_attempts:
        print(f"Failed to generate an augmented version for: {message}. Exceeded {max_attempts} attempts.")
        return None

    # Template for the prompt to be sent to the LLM
    prompt = ("""Your task is to generate a new version of the given message while retaining its original meaning.

              Example: 
              Original: "Congrats! You've won a $1000 gift card from Walmart. Click on this link to claim now: www.offer.com"
              Rephrased: "Congratulations! You have been awarded a $1000 Walmart gift card. Follow this link to redeem your prize now: www.offer.com"

              Original: "{message}"
              Rephrased:""")

    # Retry loop to handle potential exceptions and ensure successful augmentation
    while True:
        try:
            # Generate augmented message using LLM
            generated_text = client.text_generation(prompt.format(message=message), temperature=0.9, max_new_tokens=128,
                                            seed=seed)
            generated_message = extract_augmented_message(generated_text)

            if generated_message is None:
                return augment_with_LLM(message, seed + 1, attempt + 1, max_attempts)

            return generated_message

        except Exception as e:
            print(f"An exception occurred: {e}")
            time.sleep(5)  # Wait for 5 seconds before trying again
            print("Retrying...")


# Initialize the InferenceClient with an API token and model
token = config.API_TOKENS[0]
client = InferenceClient(token=token, model="meta-llama/Llama-2-70b-chat-hf")

# Load training data and filter for spam messages
data_df = pd.read_csv('./train_data.csv')

# Separate the spam messages
spam_df = data_df[data_df['label'] == 'spam']
num_variants = 2  # Number of augmented variants to generate per message

augmented_rows = []  # List to hold augmented data
total = len(spam_df)

# Iterate through the spam dataframe to generate augmented messages
for i, (idx, row) in enumerate(spam_df.iterrows()):
    message = row['message']
    print(f"Original: {message}")

    # Generate different variants of the message
    for variant in range(num_variants):
        augmented_message = augment_with_LLM(message, generate_seed(idx, variant))

        if augmented_message:
            print(f"Rephrased (Version {variant + 1}): {augmented_message}")
            augmented_rows.append([augmented_message, 'spam'])

    # Print progress after every 10 messages
    if (i + 1) % 10 == 0:
        print(f"Progress: {i + 1}/{total} ({((i + 1) / total) * 100:.2f}%)")
    print("------------------------")  # Separator for better visual clarity

# Convert the list of augmented rows into a DataFrame
df_augmented = pd.DataFrame(augmented_rows, columns=['message', 'label'])

# Save the augmented dataframe to a CSV file
output_path = './Spam/augmented_spam_2.csv'
df_augmented.to_csv(output_path, index=False)
