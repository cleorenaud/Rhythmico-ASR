import os
import csv
from collections import defaultdict, Counter
import statistics
import numpy as np

def analyze_phoneme_statistics(folder_path):
    phoneme_counts = defaultdict(int)
    phoneme_probs = defaultdict(list)
    top1_probs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    for i in range(1, 4):  # Phoneme1, Phoneme2, Phoneme3
                        phoneme = row[f"Phoneme{i}"]
                        if phoneme in ("[PAD]", "|"):  # ✅ Exclude '[PAD]' and '|'
                            continue
                        prob = float(row[f"Prob{i}"])
                        phoneme_counts[phoneme] += 1
                        phoneme_probs[phoneme].append(prob)

                    if row["Phoneme1"] not in ("[PAD]", "|"):
                        top1_probs.append(float(row["Prob1"]))

    phoneme_stats = {
        p: {
            "count": phoneme_counts[p],
            "avg_prob": statistics.mean(phoneme_probs[p]),
            "min_prob": min(phoneme_probs[p]),
            "max_prob": max(phoneme_probs[p])
        } for p in phoneme_probs
    }

    return phoneme_stats, top1_probs


def analyze_phoneme_diversity_and_top1(folder_path):
    diversity_counts = Counter()  # key = num unique phonemes per timestep
    top1_phoneme_counts = Counter()

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Exclude '|'
                    phonemes = [row[f"Phoneme{i}"] for i in range(1, 4) if row[f"Phoneme{i}"] not in ("[PAD]", "|")]
                    unique_count = len(set(phonemes))
                    diversity_counts[unique_count] += 1

                    top1 = row["Phoneme1"]
                    if top1 not in ("[PAD]", "|"):  # Exclude top1 phoneme if it's '|'
                        top1_phoneme_counts[top1] += 1

    return diversity_counts, top1_phoneme_counts