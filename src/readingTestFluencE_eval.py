import csv
import ast
from collections import deque


def eval_0(ground_truth, child_transcription):
    gt = ground_truth.strip().split()
    ct = child_transcription.strip().split()

    found_words = []  # List to store found words
    score = 0  # Initialize score
    counted_words = set()  # Set to track words already counted as correct

    # First, handle words of 4 characters or more
    for word in gt:
        if len(word) >= 4:  # Only consider words of 4 characters or more
            # Check if the word is in the child transcription and not already counted
            if word in ct and word not in counted_words:
                found_words.append(word)
                score += 1
                counted_words.add(word)  # Mark word as counted
                ct.remove(word)  # Remove the matched word from the child transcription

    # Then, handle the remaining words (less than 4 characters)
    for word in gt:
        if len(word) < 4:  # Only consider words of less than 4 characters
            # Check if the word is in the child transcription and not already counted
            if word in ct and word not in counted_words:
                found_words.append(word)
                score += 1
                counted_words.add(word)  # Mark word as counted
                ct.remove(word)  # Remove the matched word from the child transcription

    return found_words, score

def eval_1(ground_truth, child_transcription):
    gt = ground_truth.strip().split()
    ct = child_transcription.strip().split()

    word_states = []                      # To store (word, "Correct"/"Incorrect") for each ground truth word
    score = 0                             # Count of correct matches
    ct_matched = [False] * len(ct)        # Track matched child words

    # Group ground truth words by their length
    length_groups = {i: [] for i in range(1, 7)}  # 1 to 6 (6 = 6 or more)

    for word in gt:
        length = min(len(word), 6)  # Group length >= 6 into key 6
        length_groups[length].append(word)

    # Create a copy of gt sorted by length priority for matching
    # But keep track of original gt order for final word_states
    sorted_gt = []
    for length in range(6, 0, -1):
        sorted_gt.extend(length_groups[length])

    # Match in length-priority order and mark matched words
    matched_words = {}  # Map from matched word to list of matched positions in ct
    for word in sorted_gt:
        for i, ct_word in enumerate(ct):
            if ct_word == word and not ct_matched[i]:
                ct_matched[i] = True
                matched_words.setdefault(word, []).append(i)
                break

    # Now reconstruct word_states in the **original gt order**
    ct_used_positions = set()
    for word in gt:
        # Check if there's a remaining unmatched ct position for this word
        matched_pos_list = matched_words.get(word, [])
        if matched_pos_list:
            # Mark one matched position as used
            used_pos = matched_pos_list.pop(0)
            ct_used_positions.add(used_pos)
            word_states.append((word, "Correct"))
            score += 1
        else:
            word_states.append((word, "Incorrect"))

    return word_states, score

def eval_2(ground_truth, child_transcription):
    gt = ground_truth.strip().split()
    ct = child_transcription.strip().split()
    
    word_states = []  # List to store (word, state) tuples
    counted_words = set()  # Set to track words already counted as correct
    score = 0  # Initialize score
    
    # Initialize a list to track which words in the child transcription have already been matched
    ct_matched = [False] * len(ct)
    
    # Iterate over words in ground truth in original order
    for word in gt:
        found = False
        for i, ct_word in enumerate(ct):
            if ct_word == word and not ct_matched[i]:  # Match word if not already matched
                word_states.append((word, "Correct"))
                counted_words.add(word)
                ct_matched[i] = True  # Mark word as matched
                found = True
                score += 1  # Increment score for correct match
                break  # Move to next word in ground truth after finding a match
        
        if not found:  # If the word wasn't found, mark as Incorrect
            word_states.append((word, "Incorrect"))
    
    return word_states, score


def eval_3(ground_truth, child_transcription):
    """
    Evaluates phonetic transcription by attempting to reconstruct expected words
    using a sliding buffer of up to 15 phonemes from the child transcription.
    Processes phonemes one by one from the child transcription, similar to evaluate_sentence.
    
    Args:
        ground_truth: String of expected phonemes (space-separated words)
        child_transcription: String of pronounced phonemes (space-separated single phonemes)
    
    Returns:
        tuple: (word_states, score) where word_states is list of (word, state) tuples
    """
    
    def can_reconstruct_word(buffer, target_word):
        """
        Checks if the target_word can be formed using phonemes from the buffer while maintaining order.
        Returns True and a list of indices used if the word is reconstructed.
        Otherwise, returns False and an empty list.
        """
        target_phonemes = list(target_word)  # Convert word to phoneme list (each char is a phoneme)
        target_index = 0  # Tracks position in target word
        used_indices = []  # Stores the buffer indices used to match the word

        for i, phoneme in enumerate(buffer):
            if target_index < len(target_phonemes) and target_phonemes[target_index] == phoneme:
                used_indices.append(i)
                target_index += 1  # Move to next phoneme

            if target_index == len(target_phonemes):  # If all phonemes were found in order
                return True, used_indices  # Return True and the indices used

        return False, []  # Could not reconstruct the word
    
    # Parse input - create a list that we'll consume one by one
    gt_words = ground_truth.strip().split()  # Split into words
    ct_string = child_transcription.replace(' ', '')  # Remove spaces to get continuous string
    phoneme_queue = list(ct_string)  # Convert to list of individual characters (phonemes)
    
    word_states = []  # List to store (word, state) tuples
    score = 0  # Initialize score
    buffer = []  # Rolling buffer for phoneme predictions
    
    # Process each word in ground truth
    for word in gt_words:   
        word_found = False
        phonemes_tried = 0
        
        # First, check if word can be reconstructed with current buffer contents
        found, used_indices = can_reconstruct_word(buffer, word)
        if found:
            word_states.append((word, "Correct"))
            score += 1
            word_found = True
            # Remove only the phonemes that were used (up to and including the last used index)
            last_used_index = max(used_indices) if used_indices else -1
            buffer = buffer[last_used_index + 1:]
        else:
            # If buffer is already full (15 phonemes) and word not found, skip to next word
            if len(buffer) >= 15:
                word_states.append((word, "Incorrect"))
                continue
            
            # If not found with current buffer, add phonemes one by one up to limit of 15 attempts
            while phonemes_tried < 15 and phoneme_queue and not word_found:
                # Add one phoneme at a time
                if len(buffer) >= 15:
                    buffer.pop(0)  # Remove oldest phoneme to maintain buffer size limit
                
                buffer.append(phoneme_queue.pop(0))  # Take next phoneme from queue
                phonemes_tried += 1
                
                # Try to reconstruct the word again
                found, used_indices = can_reconstruct_word(buffer, word)
                if found:
                    word_states.append((word, "Correct"))
                    score += 1
                    word_found = True
                    # Remove only the phonemes that were used (up to and including the last used index)
                    last_used_index = max(used_indices) if used_indices else -1
                    buffer = buffer[last_used_index + 1:]
        
        # If word not found after trying up to 15 phonemes
        if not word_found:
            word_states.append((word, "Incorrect"))
    
    return word_states, score


def load_predictions(csv_filename):
    """
    Reads the CSV file and returns a list of sets, where each set contains the three possible phonemes per timestamp.
    """
    phoneme_options = []
    
    with open(csv_filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            _, top1, top2, top3 = row
            phoneme_set = {top1}
            if top2 != "[PAD]":
                phoneme_set.add(top2)
            if top3 != "[PAD]":
                phoneme_set.add(top3)
            
            phoneme_options.append(phoneme_set)
    
    return phoneme_options


def evaluate_sentence(phoneme_options, target_sentence, buffer_size):
    """
    Evaluates whether each word in the target sentence was correctly pronounced using a rolling buffer.
    - Adds phoneme options to a buffer one at a time.
    - Checks if the word can be formed by selecting at most one phoneme per timestamp (while keeping order).
    - If a match is found, removes only the tuples in the buffer **that were used** for the current word.
    - If the buffer is full and the word is not matched, moves to the next word without clearing the buffer.
    Returns a word-by-word evaluation.
    """
    def can_reconstruct_word(buffer, target_word):
        """
        Checks if the target_word can be formed using at most one phoneme per timestamp (while maintaining order).
        Returns True and a list of indices used if the word is reconstructed.
        Otherwise, returns False and an empty list.
        """
        target_phonemes = list(target_word)  # Convert word to phoneme list
        target_index = 0  # Tracks position in target word
        used_indices = []  # Stores the buffer indices used to match the word

        for i, phoneme_set in enumerate(buffer):
            if target_index < len(target_phonemes) and target_phonemes[target_index] in phoneme_set:
                used_indices.append(i)
                target_index += 1  # Move to next phoneme

            if target_index == len(target_phonemes):  # If all phonemes were found in order
                return True, used_indices  # Return True and the indices used

        return False, []  # Could not reconstruct the word

    words = target_sentence.split(" ")  # Split sentence into words
    buffer = []  # Rolling buffer for phoneme predictions
    results = []

    for word in words:
        print("==" * 20)
        print(f"Target word: {word}")

        for _ in range(buffer_size):  # Ensure we don't go beyond buffer size
            found, used_indices = can_reconstruct_word(buffer, word)

            if found:  # If the word is successfully reconstructed
                results.append((word, "correct"))
                
                # Remove only the used elements while keeping order
                buffer = [buffer[i] for i in range(len(buffer)) if i not in used_indices]
                break  # Move to next word
            else:
                if phoneme_options:
                    buffer.append(phoneme_options.pop(0))

                    print(f"Buffer: {buffer}")
                   
        else:  # If no match is found after buffer fills up, move on
            results.append((word, "missed"))

    return results

def top_3_phoneme_evaluation(readingTestFluencE_df, test_id, expected_phonemes, buffer_size=15, detailed=False):
    """"
    Evaluates the phoneme predictions against the ground truth using a buffer

    Args:
    - readingTestFluencE_df: DataFrame containing the test data.
    - test_id: The ID of the test to evaluate.
    - expected_phonemes: The ground truth sentence to compare against.
    - buffer_size: The size of the rolling buffer for phoneme predictions.
    - detailed: If True, prints detailed evaluation results.
    """
    test_row = readingTestFluencE_df[readingTestFluencE_df['id'] == test_id]
    test_dict = ast.literal_eval(test_row['testParameters'].values[0])
    
    selected_text = test_dict['textSelected']['text']
    selected_words = selected_text.split()
    
    evaluation_result = test_row['evaluationResults'].apply(
        lambda x: x['wordsState'] if 'wordsState' in x else None).dropna().tolist()
    
    # We extract the ground truth for each test
    read_words = [[d for d in row if list(d.values())[0] != "NonRead"] for row in evaluation_result]
    reference_text = ' '.join([list(d.keys())[0] for row in read_words for d in row])
    target_sentence = " ".join(expected_phonemes[:len(reference_text.split())]) 

    csv_filename = f"sample_readingTestFluencE/readingTestFluencE_{test_id}_phonemes.csv"

    # Load the phoneme predictions
    phoneme_options = load_predictions(csv_filename)

    # Evaluate each word
    word_results = evaluate_sentence(phoneme_options, target_sentence, buffer_size=buffer_size)

    print(f"Buffer size: {buffer_size}")

    if detailed:
        # Print detailed evaluation
        print("Detailed Evaluation:")
        for (phoneme_word, status), grapheme_word in zip(word_results, selected_words):
            status_symbol = "✅" if status == "correct" else "❌"
            print(f"{status_symbol} {grapheme_word} → {status}")

    # Summary of correctness
    correct_words = sum(1 for _, status in word_results if status == "correct")
    total_words = len(word_results)
    print(f"\nFinal Score: {correct_words}/{total_words} words read correctly.")
