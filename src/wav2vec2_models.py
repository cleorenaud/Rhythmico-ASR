import os
import csv
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor
from src.data_processing import *

def sanitize_row(row):
    """
    Replaces malformed standalone tilde characters in a CSV row.
    Specifically, replaces the combining tilde character (U+0303) with a standalone tilde (U+02DC)
    when it appears isolated.
    """
    return [col.replace("̃", "˜") if col == "̃" else col for col in row]


def top_3_phoneme_transcriptions(tests_id, folder_path):
    # We iterate over the tests_id and we create the top-3 phoneme transcriptions
    for test_id in tests_id:
        # We extract the audio file
        audio_file = f"{folder_path}_{test_id}.wav"
        audio, _ = sf.read(audio_file)

        # Preprocess the audio and prepare the inputs for the model
        inputs = processor(np.array(audio), sampling_rate=16_000., return_tensors="pt")

        # Get the top-3 most probable phonemes for each timestep
        with torch.no_grad():
            logits = model(**inputs).logits

        topk_probs, topk_indices = torch.topk(logits, k=3, dim=-1)

        # Decode the top-3 predictions for each timestep
        filtered_transcriptions = []
        last_phoneme_set = None  # We store the last added phoneme set to avoid duplicates

        for i in range(topk_indices.shape[1]):  # Iterate over time steps
            phonemes = processor.tokenizer.convert_ids_to_tokens(topk_indices[0, i].tolist())

            # Skip if the first prediction is '[PAD]' or the first two predictions are '|' and '[PAD]'
            if (phonemes[0] == "[PAD]") or (phonemes[0] == "|" and phonemes[1] == "[PAD]"):
                continue  # Ignore this timestamp

            phoneme_tuple = (phonemes[0], phonemes[1], phonemes[2])

            # Avoid adding consecutive duplicate phoneme sets
            if phoneme_tuple != last_phoneme_set:
                row = [i] + list(phoneme_tuple)
                sanitized_row = sanitize_row(row)
                filtered_transcriptions.append(sanitized_row)
                last_phoneme_set = phoneme_tuple  # Update last seen set

        # Define output CSV file name
        csv_filename = os.path.splitext(audio_file)[0] + "_phonemes.csv"

        # Save results to CSV
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestep", "Top_1", "Top_2", "Top_3"])  # CSV header
            writer.writerows(filtered_transcriptions)

        print(f"Filtered top-3 phoneme transcriptions saved to {csv_filename}")

def top_3_phoneme_transcriptions_with_proba(folder_path, test_type, output_folder, tests_id, model_id="Cnam-LMSSC/wav2vec2-french-phonemizer"):
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    # Initialize the model and processor
    MODEL_ID = model_id
    model = AutoModelForCTC.from_pretrained(MODEL_ID)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    for test_id in tests_id:
        audio_file = f"{folder_path}/{test_type}_{test_id}.wav"
        audio, _ = sf.read(audio_file)

        inputs = processor(np.array(audio), sampling_rate=16_000., return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        topk_probs, topk_indices = torch.topk(logits, k=3, dim=-1)
        topk_probs = topk_probs.softmax(dim=-1)

        filtered_transcriptions = []
        last_phoneme_set = None

        for i in range(topk_indices.shape[1]):
            phonemes = processor.tokenizer.convert_ids_to_tokens(topk_indices[0, i].tolist())
            probs = topk_probs[0, i].tolist()

            if (phonemes[0] == "[PAD]") or (phonemes[0] == "|" and phonemes[1] == "[PAD]"):
                continue

            phoneme_tuple = (phonemes[0], phonemes[1], phonemes[2])
            if phoneme_tuple != last_phoneme_set:
                row = [i]
                for p, prob in zip(phonemes, probs):
                    row.extend([p, round(prob, 4)])
                sanitized_row = sanitize_row(row)
                filtered_transcriptions.append(sanitized_row)
                last_phoneme_set = phoneme_tuple

        output_path = os.path.join(output_folder, f"{test_type}_{test_id}_phonemes.csv")
        with open(output_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Phoneme1", "Prob1", "Phoneme2", "Prob2", "Phoneme3", "Prob3"])
            writer.writerows(filtered_transcriptions)

        print(f"Filtered top-3 phoneme transcriptions saved to {output_path}")