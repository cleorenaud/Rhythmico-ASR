import numpy as np
import Levenshtein

def levenshtein_similarity(a: str, b: str) -> float:
    """
    Compute the similarity between two strings using the Levenshtein distance.
    
    Args:
        a: The first string.
        b: The second string.
    """
    if not a and not b:
        return 1.0
    distance = Levenshtein.distance(a, b)
    return 1 - distance / max(len(a), len(b))

def evaluate_exact_word_match(ground_truth_phonemes: str, child_transcription: str):
    """
    Evaluate the child transcription using exact word match.
    
    Args:
        ground_truth_phonemes: The ground truth phonemes.
        child_transcription: The child transcription.
    """
    gt_words = ground_truth_phonemes.strip().split()
    child_words = child_transcription.strip().split()

    correct_words = sum(1 for gt, ct in zip(gt_words, child_words) if gt == ct)

    return {
        'method': 'Exact Word Match',
        'score': correct_words
    }

def evaluate_levenshtein_word_match(ground_truth_phonemes: str, child_transcription: str, threshold: float = 0.5):
    """
    Evaluate the child transcription using Levenshtein word match.
    
    Args:
        ground_truth_phonemes: The ground truth phonemes.
        child_transcription: The child transcription.
        threshold: The similarity threshold
    """
    gt_words = ground_truth_phonemes.strip().split()
    child_words = child_transcription.strip().split()

    correct_words = 0

    for gt, ct in zip(gt_words, child_words):
        similarity = levenshtein_similarity(gt, ct)
        if similarity >= threshold:
            correct_words += 1

    return {
        'method': 'Levenshtein Word Match',
        'score': correct_words
    }

def evaluate_chunked_alignment(ground_truth_phonemes: str, child_transcription: str, threshold: float = 0.8):
    """
    Evaluate the child transcription using chunked alignment.
    
    Args:
        ground_truth_phonemes: The ground truth phonemes.
        child_transcription: The child transcription.
        threshold: The similarity threshold
    """
    gt_words = ground_truth_phonemes.strip().split()
    child_phonemes = child_transcription.replace(" ", "")

    correct_words = 0
    child_index = 0

    for gt in gt_words:
        gt_len = len(gt)
        if child_index >= len(child_phonemes):
            break

        candidate = child_phonemes[child_index:child_index + gt_len]
        similarity = levenshtein_similarity(gt, candidate)

        if similarity >= threshold:
            correct_words += 1

        child_index += gt_len

    return {
        'method': 'Chunked Alignment',
        'threshold': threshold,
        'score': correct_words
    }

def evaluate_sliding_window(ground_truth_phonemes: str, child_transcription: str, window_size: int = 5, threshold: float = 0.8):
    """
    Evaluate the child transcription using sliding window.
    
    Args:
        ground_truth_phonemes: The ground truth phonemes.
        child_transcription: The child transcription.
        window_size: The window size.
        threshold: The similarity threshold
    """
    gt_words = ground_truth_phonemes.strip().split()
    child_phonemes = child_transcription.replace(" ", "")

    correct_words = 0
    child_index = 0

    for gt in gt_words:
        gt_len = len(gt)
        found = False

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
                child_index += 1

        if not found:
            pass

    return {
        'method': 'Sliding Window',
        'window_size': window_size,
        'threshold': threshold,
        'score': correct_words
    }

def compare_methods_with_different_parameters(ground_truth_phonemes: str, child_transcription: str, window_sizes: list, thresholds: list):
    """
    Helper function to run the evaluate_sliding_window method with different window sizes and thresholds.

    Args:
        ground_truth_phonemes: The ground truth phonemes.
        child_transcription: The child transcription.
        window_sizes: The window sizes.
        thresholds: The thresholds
    """
    results = []
    
    # Iterate through each combination of window_size and threshold
    for window_size in window_sizes:
        for threshold in thresholds:
            result = evaluate_sliding_window(ground_truth_phonemes, child_transcription, window_size, threshold)
            results.append(result)
    
    # Now print or return the results to compare
    return results

def compare_methods_with_different_thresholds(ground_truth_phonemes: str, child_transcription: str, thresholds: list):
    """
    Helper function to run the evaluate_chunked_alignment method with different thresholds.
    
    Args:
        ground_truth_phonemes: The ground truth phonemes.
        child_transcription: The child transcription.
        thresholds: The thresholds
    """
    results = []
    
    # Iterate through each threshold value
    for threshold in thresholds:
        result = evaluate_chunked_alignment(ground_truth_phonemes, child_transcription, threshold)
        results.append(result)
    
    return results

def run_all_evaluations(ground_truth_phonemes: str, child_transcription: str):
    """
    Run all evaluation methods.
    
    Args:
        ground_truth_phonemes: The ground truth phonemes.
        child_transcription: The child transcription
    """
    methods = [
        evaluate_exact_word_match,
        evaluate_levenshtein_word_match,
        evaluate_chunked_alignment,
        evaluate_sliding_window,
    ]

    results = []
    for method in methods:
        if 'threshold' in method.__code__.co_varnames:
            result = method(ground_truth_phonemes, child_transcription, threshold=0.8)
        else:
            result = method(ground_truth_phonemes, child_transcription)
        results.append(result)

    return results


