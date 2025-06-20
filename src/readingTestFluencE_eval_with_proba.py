import csv
import ast

def load_predictions_with_proba(csv_filename, proba_threshold=0.1):
    """
    Reads the CSV file and returns a list of tuples, where each tuple is:
        (timestamp, predictions)
    'predictions' is a list of (phoneme, probability) tuples.
    Only phonemes with probability >= proba_threshold (default 10%) are included.
    
    Expected CSV format per row:
       timestamp, phoneme1, proba1, phoneme2, proba2, phoneme3, proba3
    """
    phoneme_options = []
    
    with open(csv_filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 7:
                continue  # Skip malformed rows

            timestamp = row[0]
            phon1, proba1, phon2, proba2, phon3, proba3 = row[1:7]
            predictions = []
            if phon1 != "[PAD]" and float(proba1) >= proba_threshold:
                predictions.append((phon1, float(proba1)))
            if phon2 != "[PAD]" and float(proba2) >= proba_threshold:
                predictions.append((phon2, float(proba2)))
            if phon3 != "[PAD]" and float(proba3) >= proba_threshold:
                predictions.append((phon3, float(proba3)))
            
            phoneme_options.append((timestamp, predictions))
    
    return phoneme_options


def can_reconstruct_word(buffer, target_word):
    """
    Tries to reconstruct target_word (a string of phoneme tokens) from the buffer.
    
    The buffer is a list of tuples (timestamp, predictions) where predictions is a list 
    of tuples (phoneme, probability).
    
    Returns:
      (found, used_indices, used_probs)
      - found is True if all phonemes in target_word are found in order.
      - used_indices: list of indices (into the current buffer) where a match was found.
      - used_probs: list of probabilities corresponding to the matched phonemes.
    """
    target_phonemes = list(target_word)
    target_index = 0
    used_indices = []
    used_probs = []
    
    for i, (timestamp, predictions) in enumerate(buffer):
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

    Returns:
        - percentage_matched: how many phonemes were matched over the word length
        - used_indices: indices in the buffer that were used
        - used_probs: probabilities of matched phonemes
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


def evaluate_sentence_with_proba(phoneme_options_with_proba, target_sentence, proba_threshold=0.3):
    """
    Evaluates each word in the target_sentence using a rolling buffer of phoneme predictions 
    (with probabilities) and dynamic buffer sizes. Instead of adding phonemes one by one, 
    it adds a chunk of phonemes corresponding to word length + extra.

    Args:
        phoneme_options_with_proba: List of tuples (timestamp, predictions) where predictions is a list of (phoneme, probability).
        target_sentence: The sentence to evaluate.
        proba_threshold: Probability threshold for phoneme inclusion.
    """
    words = target_sentence.split()
    buffer = []
    results = []
    
    for word in words:
        # TODO: remove
        # print("==" * 20)
        # print(f"Evaluating word: {word}")

        matched = False

        # We add the number of phonemes in the word to the buffer
        for _ in range(len(word)):
            if phoneme_options_with_proba:
                buffer.append(phoneme_options_with_proba.pop(0))
            else:
                break

        # # We remove old phonemes from the buffer if they are too far from the latest timestamp
        # if buffer:
        #     latest_timestamp = int(buffer[-1][0])
        #     buffer = [item for item in buffer if abs(int(item[0]) - latest_timestamp) < 100]

        # Try matching the word
        found, used_indices, used_probs = can_reconstruct_word(buffer, word)

        # TODO: remove
        # print(f"Buffer: {buffer}")

        if found:
            if any(prob < proba_threshold for prob in used_probs):
                state = "uncertain_correct"
            else:
                state = "correct"

            timestamp = buffer[used_indices[0]][0] if used_indices else None
            results.append((word, state, timestamp))
            
            last_used_index = used_indices[-1]
            buffer = buffer[last_used_index + 1:]  # remove used items
            matched = True
        else:
            # Try partial match
            percentage, used_indices, used_probs = partial_match_percentage(buffer, word)
            if percentage >= 0.4:
                state = "uncertain_incorrect"

                timestamp = buffer[used_indices[0]][0] if used_indices else None
                results.append((word, state, timestamp))
                
                # last_used_index = used_indices[-1] if used_indices else 0
                # buffer = buffer[last_used_index + 1:]
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

    print(f"\n❌ False Positives: {false_positives}")
    print(f"❌ False Negatives: {false_negatives}")
    return false_positives, false_negatives


def top_3_phoneme_evaluation_with_proba(readingTest_df, test_type, test_id, expected_phonemes, detailed=False):
    """
    Evaluates the phoneme predictions against the ground truth using a rolling buffer that takes into account probabilities.
    
    For each word, returns a tuple (word, state, timestamp), where:
      - state is 'correct', 'incorrect', or 'uncertain'
      - timestamp (if provided) is the timestamp at which the word began (only returned if the word is uncertain)
    
    Args:
      readingTestFluencE_df: DataFrame containing the test data.
      test_id: The test ID.
      expected_phonemes: The ground truth sentence in phoneme representation.
      detailed: If True, prints detailed evaluation.
    """
    test_row = readingTest_df[readingTest_df['id'] == test_id]
    test_dict = ast.literal_eval(test_row['testParameters'].values[0])
    
    selected_text = test_dict['textSelected']['text']  # Grapheme (alphabetical) ground truth
    selected_words = selected_text.split()
    
    evaluation_result = test_row['evaluationResults'].apply(
        lambda x: x['wordsState'] if 'wordsState' in x else None).dropna().tolist()
        
    # Extract ground truth (phoneme representation) from evaluation results.
    read_words = [[d for d in row if list(d.values())[0] != "NonRead"] for row in evaluation_result]
    reference_text = ' '.join([list(d.keys())[0] for row in read_words for d in row])
    # Ensure we have as many expected_phoneme words as there are in the reference text.
    target_sentence = " ".join(expected_phonemes[:len(reference_text.split())])
    
    csv_filename = f"sample_{test_type}/{test_type}_{test_id}_phonemes.csv"
    # Load phoneme predictions with probability data
    phoneme_options_with_proba = load_predictions_with_proba(csv_filename)
    
    # Evaluate the sentence using the updated function.
    word_results = evaluate_sentence_with_proba(phoneme_options_with_proba, target_sentence)

    false_pos, false_neg = compare_evaluations(word_results, evaluation_result) 
    
    if detailed:
        print("Detailed Evaluation:")
        for (word, state, ts), grapheme_word in zip(word_results, selected_words):
            if state == "correct":
                symbol = "✅"
            elif state == "incorrect":
                symbol = "❌"
            elif state == "uncertain_correct":
                symbol = "⚠️✅"
            elif state == "uncertain_incorrect":
                symbol = "⚠️❌"
            else:
                symbol = "❓"

            if state == "uncertain":
                print(f"{symbol} {grapheme_word} (starts at {ts}) → {state}")
            else:
                print(f"{symbol} {grapheme_word} → {state}")
    
    correct_words = sum(1 for _, state, _ in word_results if state=="correct")
    total_words = len(word_results)
    print(f"\nFinal Score: {correct_words}/{total_words} words read correctly.")

    # We print the number of words for each state
    state_counts = {}
    for _, state, _ in word_results:
        if state not in state_counts:
            state_counts[state] = 0
        state_counts[state] += 1
    print("\nState Counts:")
    for state, count in state_counts.items():
        print(f"{state}: {count}")

    return word_results, false_pos, false_neg
