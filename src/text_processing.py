import os
import ast
import pandas as pd
from phonemizer import phonemize

def save_phonetic_transcription_to_csv(text, test_type, folder="transcriptions", file_name="test_phonetic_transcriptions.csv"):
    """
    Function to save the phonetic transcription of a text to a CSV file.

    Args:
    text (str): The text to phonemize and save.
    test_type (str): The type of test the text is from.
    folder (str): The folder where the CSV file will be saved.
    file_name (str): The name of the CSV file.
    """
    # Ensure the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Step 3: Phonemize the testParameters text
    try:
        phonetic_transcription = phonemize(
            [text],
            language='fr-fr',
            backend='espeak',
            strip=True,
            njobs=1
        )[0]  # We use the [0] to get the first (and only) result
    except Exception as e:
        print(f"Error during phonemization: {e}")
        return
    
    # Step 4: Prepare data for CSV
    output_data = {
        "Test Type": [test_type],  # Replace with the actual test type if needed
        "Phonetic Transcription": [phonetic_transcription]
    }
    
    # Combine folder and file name to get the full path
    file_path = os.path.join(folder, file_name)
    
    # Step 5: Write to CSV (if file exists, append to it; if not, create it)
    df_output = pd.DataFrame(output_data)
    df_output.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

    print(f"Phonetic transcription saved to {file_path}")
