import os
import csv
from collections import defaultdict
import statistics

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
                        if phoneme == "[PAD]":
                            continue
                        prob = float(row[f"Prob{i}"])
                        phoneme_counts[phoneme] += 1
                        phoneme_probs[phoneme].append(prob)

                    # Track probability of most probable phoneme (Phoneme1)
                    if row["Phoneme1"] != "[PAD]":
                        top1_probs.append(float(row["Prob1"]))

    # Prepare structured statistics
    phoneme_stats = {}
    for phoneme, probs in phoneme_probs.items():
        phoneme_stats[phoneme] = {
            "count": phoneme_counts[phoneme],
            "avg_prob": statistics.mean(probs),
            "min_prob": min(probs),
            "max_prob": max(probs)
        }

    top1_prob_stats = {
        "avg_prob": statistics.mean(top1_probs),
        "min_prob": min(top1_probs),
        "max_prob": max(top1_probs)
    }

    return phoneme_stats, top1_prob_stats
