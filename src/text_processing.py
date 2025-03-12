import os
import pandas as pd
from phonemizer import phonemize

# Define a function to extract the text, phonemize it, and save to a CSV in the transcriptions folder
def save_phonetic_transcription_to_csv(data, test_id, folder="transcriptions", file_name="test_phonetic_transcriptions.csv"):
    # Ensure the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Step 1: Extract the row with the given test_id
    row = data[data['id'] == test_id]
    
    if row.empty:
        print(f"No row found for id: {test_id}")
        return
    
    # Step 2: Extract the testParameters text from the row
    testParameters = row[['testParameters']].values[0][0]  # Extract the text value from the row
    
    if not isinstance(testParameters, str):
        print(f"testParameters is not a string: {testParameters}")
        return
    
    # Step 3: Phonemize the testParameters text
    try:
        phonetic_transcription = phonemize(
            [testParameters],
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
        "Test Type": ["readingTestFluencE"],  # Replace with the actual test type if needed
        "Phonetic Transcription": [phonetic_transcription]
    }
    
    # Combine folder and file name to get the full path
    file_path = os.path.join(folder, file_name)
    
    # Step 5: Write to CSV (if file exists, append to it; if not, create it)
    df_output = pd.DataFrame(output_data)
    df_output.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

    print(f"Phonetic transcription saved to {file_path}")
