import time
import pandas as pd
from huggingface_hub import InferenceClient

import config

# Import API tokens
api_tokens = config.API_TOKENS


def evaluate_model(data_df, prompt_template, model_name, output_file):
    # Initialize the index to track processed entries
    processed_index = 0

    # Define the number of sentences to process per batch
    entries_per_iteration = 1000

    # Process all entries using different tokens
    for token in api_tokens:
        if processed_index >= len(data_df):
            break

        client = InferenceClient(token=token, model=model_name)

        if processed_index + entries_per_iteration < len(data_df):
            batch = data_df.iloc[processed_index:processed_index + entries_per_iteration]
        else:
            batch = data_df.iloc[processed_index:len(data_df)]

        batch_predictions = []
        for index, row in batch.iterrows():
            message = row["message"]
            actual_label = row["label"]

            # Predict label
            predicted_label = None
            retry = True
            while retry:
                try:
                    predicted_label = client.text_generation(prompt_template.format(message=message), max_new_tokens=3,
                                                             temperature=0.1)
                    retry = False  # If success, exit the retry loop
                except Exception as e:
                    print(f"An exception occurred: {e}")
                    time.sleep(5)  # Wait for 5 seconds before trying again
                    print("Retrying...")

            batch_predictions.append({
                "message": message,
                "predicted_label": predicted_label,
                "actual_label": actual_label
            })

            # Print the labels to the console
            print(f"Message: {message}")
            print(f"Predicted Label: {predicted_label}")
            print(f"Actual Label: {actual_label}")
            print("-"*30)

        # Save batch predictions to a file
        batch_predictions_df = pd.DataFrame(batch_predictions)
        batch_predictions_df.to_csv(output_file, mode='a', header=(processed_index == 0), index=False)

        print(f"Token {token} - Processed {processed_index + len(batch_predictions)} out of {len(data_df)} SMS.")

        processed_index += entries_per_iteration  # Move to the next batch of entries

    # Print a completion message
    print(f"All entries processed and predictions saved for {model_name}.")
