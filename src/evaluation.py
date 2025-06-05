import ast
import csv


def load_predictions(csv_filename, proba_threshold=0.1):
    """
    Reads the CSV file and returns a list of tuples (timestamp, predictions) where 'predictions' is a list of (phoneme, probability) tuples.
    
    Args:
        csv_filename: Path to the CSV file containing phoneme predictions.
        proba_threshold: Probability threshold for phoneme inclusion.

    Returns:
        List of tuples (timestamp, predictions) where 'predictions' is a list of (phoneme, probability) tuples.
    
    Expected CSV format per row:
       timestamp, phoneme1, proba1, phoneme2, proba2, phoneme3, proba3.
    """
    phoneme_options = []
    skipped_phonemes = ['[PAD]', '|'] # List of phonemes that we discard
    
    with open(csv_filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            if len(row) < 7:
                continue

            timestamp = row[0]
            phon1, proba1, phon2, proba2, phon3, proba3 = row[1:7]
            predictions = []

            # We skip the phonemes that are in the skipped_phonemes list or are below the probability threshold
            if phon1 not in skipped_phonemes and float(proba1) >= proba_threshold:
                predictions.append((phon1, float(proba1)))
            if phon2 not in skipped_phonemes and float(proba2) >= proba_threshold:
                predictions.append((phon2, float(proba2)))
            if phon3 not in skipped_phonemes and float(proba3) >= proba_threshold:
                predictions.append((phon3, float(proba3)))
            
            phoneme_options.append((timestamp, predictions))
    
    return phoneme_options


def can_reconstruct_word(buffer, target_word):
    """
    Tries to reconstruct target_word (a string of phoneme tokens) using the phonemes in the buffer.
    
    Args:
        buffer: List of tuples (timestamp, predictions) where predictions is a list of (phoneme, probability).
        target_word: The target word to reconstruct.

    Returns:
        - found: True if the word can be reconstructed, False otherwise.
        - used_indices: Indices of the phonemes in the buffer that were used to reconstruct the word.
        - used_probs: Probabilities of matched phonemes.
    """
    target_phonemes = list(target_word)
    target_index = 0
    used_indices = []
    used_probs = []
    
    for i, (_, predictions) in enumerate(buffer):
        if target_index < len(target_phonemes):
            # Look for a prediction matching the next target phoneme exactly.
            for phoneme, prob in predictions:
                if phoneme == target_phonemes[target_index]:
                    used_indices.append(i)
                    used_probs.append(prob)
                    target_index += 1
                    break  # Move on to match the next target phoneme.
        if target_index == len(target_phonemes):
            return True, used_indices, used_probs
    
    return False, [], []


def partial_match_percentage(buffer, target_word):
    """
    Finds how many phonemes from the target word were correctly matched in order
    from the buffer (predicted phonemes with timestamps and probabilities).

    Args:
        buffer: List of tuples (timestamp, predictions) where predictions is a list of (phoneme, probability).
        target_word: The target word to match.

    Returns:
        - percentage: The percentage of phonemes matched.
        - used_indices: Indices of the phonemes in the buffer that were used to reconstruct the word.
        - used_probs: Probabilities of matched phonemes.
    """
    target_phonemes = list(target_word)
    buffer_len = len(buffer)
    word_len = len(target_phonemes)

    # Dynamic programming table: dp[i][j] = (match count, path)
    dp = [[(0, []) for _ in range(buffer_len + 1)] for _ in range(word_len + 1)]

    for i in range(1, word_len + 1):
        for j in range(1, buffer_len + 1):
            phoneme_matches = False
            match_prob = None

            for phoneme, prob in buffer[j - 1][1]:  # buffer[j-1] is (timestamp, predictions)
                if phoneme == target_phonemes[i - 1]:
                    phoneme_matches = True
                    match_prob = prob
                    break

            if phoneme_matches:
                # Match found: increment match count and store path
                prev_count, prev_path = dp[i - 1][j - 1]
                dp[i][j] = (prev_count + 1, prev_path + [(j - 1, match_prob)])
            else:
                # No match: carry forward the better of top or left
                top_count, top_path = dp[i - 1][j]
                left_count, left_path = dp[i][j - 1]
                if top_count >= left_count:
                    dp[i][j] = (top_count, top_path)
                else:
                    dp[i][j] = (left_count, left_path)

    # Best match info is in dp[word_len][buffer_len]
    match_count, match_info = dp[word_len][buffer_len]
    used_indices = [i for i, _ in match_info]
    used_probs = [prob for _, prob in match_info]
    percentage = match_count / word_len

    return percentage, used_indices, used_probs


def evaluate_sentence(phoneme_options, target_sentence, correct_threshold=0.3, incorrect_threshold=0.4):
    """
    Evaluates whether each word was correctly said using a rolling buffer of phoneme predictions and dynamic buffer sizes. 

    Args:
        phoneme_options: List of tuples (timestamp, predictions) where predictions is a list of (phoneme, probability).
        target_sentence: The target sentence to evaluate.
        correct_threshold: If one of the phonemes in the word has a probability below this threshold, the evaluation is considered uncertain.
        incorrect_threshold: If the percentage of phonemes matched is below this threshold, the evaluation is considered uncertain.

    Returns:
        - results: List of tuples (word, state, timestamp) where:
            - word: The word being evaluated.
            - state: 'correct', 'incorrect', 'uncertain_correct', or 'uncertain_incorrect'.
            - timestamp: The timestamp at which the word began (only returned if the word is uncertain).
    """
    words = target_sentence.split()
    buffer = []
    results = []
    
    for word in words:
        matched = False

        # We add the number of phonemes in the word to the buffer
        for _ in range(len(word)):
            if phoneme_options:
                buffer.append(phoneme_options.pop(0))
            else:
                break

        # Try matching the word
        found, used_indices, used_probs = can_reconstruct_word(buffer, word)

        # If we were able to reconstruct the word we check the probabilities
        if found:
            if any(prob < correct_threshold for prob in used_probs):
                state = "uncertain_correct"
            else:
                state = "correct"

            timestamp = buffer[used_indices[0]][0] if used_indices else None
            results.append((word, state, timestamp))
            
            # We remove the phonemes up to the last used index
            last_used_index = used_indices[-1]
            buffer = buffer[last_used_index + 1:]  
            matched = True
        else:
            # Otherwise we check if we can partially match the word
            percentage, used_indices, used_probs = partial_match_percentage(buffer, word)
            if percentage >= incorrect_threshold:
                state = "uncertain_incorrect"

                timestamp = buffer[used_indices[0]][0] if used_indices else None
                results.append((word, state, timestamp))
                
                matched = True

        if not matched:
            results.append((word, "incorrect", None))

    return results


def compare_evaluations(word_results, evaluation_result):
    """
    Compares the computed evaluation (`word_results`) to the original evaluationResult (`evaluation_result`),
    and computes the number of false positives and false negatives.

    Args:
      word_results: List of (word, state, timestamp) from computed phoneme evaluation.
      evaluation_result: Nested list of dicts, each row being the original word states.

    Returns:
      false_positives: Number of words wrongly predicted as correct.
      false_negatives: Number of words wrongly predicted as incorrect.
    """

    # Flatten original evaluation (exclude NonRead)
    original = [list(d.values())[0].lower() for row in evaluation_result for d in row if list(d.values())[0] != "NonRead"]
    predicted = [state.lower() for _, state, _ in word_results]

    # Make sure lengths match
    if len(original) != len(predicted):
        print(f"Length mismatch! Original: {len(original)}, Predicted: {len(predicted)}")
        return None

    false_positives = 0
    false_negatives = 0

    for orig, pred in zip(original, predicted):
        if pred == "correct" and orig != "correct":
            false_positives += 1
        elif pred == "incorrect" and orig == "correct":
            false_negatives += 1

    return false_positives, false_negatives


def evaluation_readingTest(readingTest_df, csv_file_path, test_id, expected_phonemes, correct_threshold, incorrect_threshold, details=False):
    """
    Evaluates the phoneme predictions against the ground truth using a rolling buffer that takes into account probabilities.
    
    Args:
        readingTest_df: DataFrame containing the reading test data.
        csv_file_path: Path to the CSV file containing phoneme predictions.
        test_id: ID of the test to evaluate.
        expected_phonemes: List of expected phonemes for the test.
        correct_threshold: Probability threshold for a phoneme to be considered correct.
        incorrect_threshold: Probability threshold for a phoneme to be considered incorrect.
        details: If True, prints detailed evaluation results.

    Returns:
        - word_results: List of tuples (word, state, timestamp) where:
            - word: The word being evaluated.
            - state: 'correct', 'incorrect', 'uncertain_correct', or 'uncertain_incorrect'.
            - timestamp: The timestamp at which the word began (only returned if the word is uncertain).
        - false_positives: Number of words wrongly predicted as correct.
        - false_negatives: Number of words wrongly predicted as incorrect.
    """
    test_row = readingTest_df[readingTest_df['id'] == test_id]
    
    evaluation_result = test_row['evaluationResults'].apply(
        lambda x: x['wordsState'] if 'wordsState' in x else None).dropna().tolist()
        
    # Extract ground truth (phoneme representation) from evaluation results.
    read_words = [[d for d in row if list(d.values())[0] != "NonRead"] for row in evaluation_result]
    reference_text = ' '.join([list(d.keys())[0] for row in read_words for d in row])
    # Ensure we have as many expected_phoneme words as there are in the reference text.
    target_sentence = " ".join(expected_phonemes[:len(reference_text.split())])
    
    # Load phoneme predictions with probability data
    phoneme_options = load_predictions(csv_file_path, proba_threshold=0.1)
    
    # Evaluate the sentence using the updated function.
    word_results = evaluate_sentence(phoneme_options, target_sentence, correct_threshold, incorrect_threshold)

    false_pos, false_neg = compare_evaluations(word_results, evaluation_result) 
    
    correct_words = sum(1 for _, state, _ in word_results if state=="correct")
    total_words = len(word_results)
    if details:
        print(f"\nFinal Score: {correct_words}/{total_words} words read correctly.")

    # We print the number of words for each state
    state_counts = {}
    for _, state, _ in word_results:
        if state not in state_counts:
            state_counts[state] = 0
        state_counts[state] += 1
    if details:
        print("\nState Counts:")
        for state, count in state_counts.items():
            print(f"{state}: {count}")

    return word_results, false_pos, false_neg
