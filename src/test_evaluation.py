import numpy as np
import Levenshtein

def levenshtein_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    distance = Levenshtein.distance(a, b)
    return 1 - distance / max(len(a), len(b))

def evaluate_exact_word_match(ground_truth_phonemes: str, child_transcription: str):
    gt_words = ground_truth_phonemes.strip().split()
    child_words = child_transcription.strip().split()

    total_words = len(gt_words)
    correct_words = sum(1 for gt, ct in zip(gt_words, child_words) if gt == ct)

    score = correct_words / total_words if total_words > 0 else 0.0

    return {
        'method': 'Exact Word Match',
        'correct': correct_words,
        'total': total_words,
        'score': score
    }

def evaluate_levenshtein_word_match(ground_truth_phonemes: str, child_transcription: str, threshold: float = 0.8):
    gt_words = ground_truth_phonemes.strip().split()
    child_words = child_transcription.strip().split()

    total_words = len(gt_words)
    correct_words = 0

    for gt, ct in zip(gt_words, child_words):
        similarity = levenshtein_similarity(gt, ct)
        if similarity >= threshold:
            correct_words += 1

    score = correct_words / total_words if total_words > 0 else 0.0

    return {
        'method': 'Levenshtein Word Match',
        'correct': correct_words,
        'total': total_words,
        'score': score
    }

def evaluate_chunked_alignment(ground_truth_phonemes: str, child_transcription: str, threshold: float = 0.8):
    gt_words = ground_truth_phonemes.strip().split()
    child_phonemes = child_transcription.replace(" ", "")

    total_words = len(gt_words)
    correct_words = 0
    child_index = 0

    for gt in gt_words:
        gt_len = len(gt)
        if child_index >= len(child_phonemes):
            break  # Out of data

        candidate = child_phonemes[child_index:child_index + gt_len]
        similarity = levenshtein_similarity(gt, candidate)

        if similarity >= threshold:
            correct_words += 1

        child_index += gt_len  # Move to next chunk

    score = correct_words / total_words if total_words > 0 else 0.0

    return {
        'method': 'Chunked Alignment',
        'correct': correct_words,
        'total': total_words,
        'score': score
    }

def evaluate_sliding_window(ground_truth_phonemes: str, child_transcription: str, window_size: int = 5, threshold: float = 0.8):
    gt_words = ground_truth_phonemes.strip().split()
    child_phonemes = child_transcription.replace(" ", "")

    total_words = len(gt_words)
    correct_words = 0
    child_index = 0

    for gt in gt_words:
        gt_len = len(gt)
        found = False

        # Slide through child's transcription
        while child_index <= len(child_phonemes) - gt_len:
            window = child_phonemes[child_index:child_index + gt_len + window_size]

            best_match = None
            best_score = 0.0

            for start in range(len(window) - gt_len + 1):
                candidate = window[start:start + gt_len]
                similarity = levenshtein_similarity(gt, candidate)

                if similarity > best_score:
                    best_score = similarity
                    best_match = candidate

            if best_score >= threshold:
                correct_words += 1
                child_index += len(best_match) if best_match else gt_len
                found = True
                break
            else:
                child_index += 1  # Move forward by one

        if not found:
            pass  # No match found, move to next gt word

    score = correct_words / total_words if total_words > 0 else 0.0

    return {
        'method': 'Sliding Window',
        'correct': correct_words,
        'total': total_words,
        'score': score
    }

def evaluate_global_similarity(ground_truth_phonemes: str, child_transcription: str):
    gt_seq = ground_truth_phonemes.replace(" ", "")
    child_seq = child_transcription.replace(" ", "")

    similarity = levenshtein_similarity(gt_seq, child_seq)

    return {
        'method': 'Global Similarity',
        'correct': int(similarity * len(gt_seq)),
        'total': len(gt_seq),
        'score': similarity
    }

def run_all_evaluations(ground_truth_phonemes: str, child_transcription: str):
    methods = [
        evaluate_exact_word_match,
        evaluate_levenshtein_word_match,
        evaluate_chunked_alignment,
        evaluate_sliding_window,
        evaluate_global_similarity
    ]

    results = []
    for method in methods:
        if 'threshold' in method.__code__.co_varnames:
            result = method(ground_truth_phonemes, child_transcription, threshold=0.8)
        else:
            result = method(ground_truth_phonemes, child_transcription)
        results.append(result)

    return results

