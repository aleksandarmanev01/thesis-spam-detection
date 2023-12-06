import time
import re
import random
import pandas as pd

import config
from huggingface_hub import InferenceClient


# Function to generate a consistent seed value based on input k
def generate_seed(k):
    base_seed = 42
    return base_seed * (k + 1)


# Function to extract a message from a text string using regex, looking for quoted text
def extract_augmented_message(text):
    match = re.search(r"\"(.*?)\"", text)
    if match:
        return match.group(1)
    else:
        return None


# Function to augment a message using a large language model
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
                return augment_with_LLM(message, seed+1, attempt+1, max_attempts)

            return generated_message

        except Exception as e:
            print(f"An exception occurred: {e}")
            time.sleep(5)  # Wait for 5 seconds before trying again
            print("Retrying...")


# Function to augment a set of messages and save them in a DataFrame
def augment_and_save(indices, class_label):
    augmented_rows = []
    print(f"Augmenting data for class {class_label}...")
    total = len(indices)
    for i, idx in enumerate(indices):
        message = data_df.loc[idx, 'message']
        augmented_message = augment_with_LLM(message, generate_seed(idx))

        # Skip if the augmented message is None
        if augmented_message is None:
            continue

        # Log the generated version for clarity
        print(f"Original: {message}")
        print(f"Rephrased: {augmented_message}")
        print("------------------------")  # Separator

        augmented_rows.append([augmented_message, class_label])

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{total} ({((i + 1) / total) * 100:.2f}%)")

    return pd.DataFrame(augmented_rows, columns=['message', 'label'])


# Initialize the InferenceClient with an API token and model
token = config.API_TOKENS[0]
client = InferenceClient(token=token, model="meta-llama/Llama-2-70b-chat-hf")

# Load training data that has to be augmented
data_df = pd.read_csv('./train_data.csv')

# Separate data into 'spam' and 'ham' categories
df_spam = data_df[data_df['label'] == 'spam']
df_ham = data_df[data_df['label'] == 'ham']

# Create an empty dataframe to store the rephrased versions
df_augmented = pd.DataFrame(data=[], columns=['message', 'label'])

# Determine the number of spam and ham messages to augment
increment_size_spam = int(len(df_spam) * 0.5)
increment_size_ham = int(len(df_ham) * 0.5)

random.seed(42)
# Generate random indices for spam and ham messages to be augmented
random_spam_indices = random.sample(list(df_spam.index), increment_size_spam)
random_ham_indices = random.sample(list(df_ham.index), increment_size_ham)

# Augment messages and add them to the augmented DataFrame
df_augmented = pd.concat([df_augmented, augment_and_save(random_spam_indices, 'spam')], ignore_index=True)
df_augmented = pd.concat([df_augmented, augment_and_save(random_ham_indices, 'ham')], ignore_index=True)

# Save the augmented dataset to a file
df_augmented.to_csv('./Mixed/augmented_mixed_50.csv', index=False)
