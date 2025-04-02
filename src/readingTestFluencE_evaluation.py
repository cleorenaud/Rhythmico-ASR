import csv

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

    found_words = []  # List to store found words
    score = 0  # Initialize score
    counted_words = set()  # Set to track words already counted as correct

    # Group ground truth words by their length
    length_groups = {i: [] for i in range(1, 7)}  # Group words by length (1 to 6 characters)

    for word in gt:
        if len(word) >= 6:  # First, group words of length 6 or more
            length_groups[6].append(word)
        elif len(word) == 5:
            length_groups[5].append(word)
        elif len(word) == 4:
            length_groups[4].append(word)
        elif len(word) == 3:
            length_groups[3].append(word)
        elif len(word) == 2:
            length_groups[2].append(word)
        else:
            length_groups[1].append(word)

    # Initialize a list to track which words in the child transcription have already been matched
    ct_matched = [False] * len(ct)

    # First pass: Check words of length 6 and more
    for length in range(6, 0, -1):
        for word in length_groups[length]:
            for i, ct_word in enumerate(ct):
                if ct_word == word and not ct_matched[i]:  # Match word if not already matched
                    found_words.append(word)
                    score += 1
                    ct_matched[i] = True  # Mark word as matched
                    break  # Move on to the next word in ground truth after finding a match

    return found_words, score

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


def evaluate_sentence(phoneme_options, target_sentence, buffer_size=15):
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

        # print(f"Buffer: {buffer}")

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
        # print("==" * 20)
        # print(f"Target word: {word}")

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

        else:  # If no match is found after buffer fills up, move on
            results.append((word, "missed"))

    return results

def top_3_phoneme_evaluation(readingTestFluencE_df, test_id, ground_truth):
    """"
    Evaluates the phoneme predictions against the ground truth using a buffer
    """
    test_row = readingTestFluencE_df[readingTestFluencE_df['id'] == test_id]
    evaluation_result = test_row['evaluationResults'].apply(
        lambda x: x['wordsState'] if 'wordsState' in x else None).dropna().tolist()

    # We extract the ground truth for each test
    read_words = [[d for d in row if list(d.values())[0] != "NonRead"] for row in evaluation_result]
    reference_text = ' '.join([list(d.keys())[0] for row in read_words for d in row])
    target_sentence = " ".join(ground_truth[:len(reference_text.split())]) 

    csv_filename = f"sample_readingTestFluencE/readingTestFluencE_{test_id}_phonemes.csv"

    # Load the phoneme predictions
    phoneme_options = load_predictions(csv_filename)

    # Evaluate each word
    word_results = evaluate_sentence(phoneme_options, target_sentence, buffer_size=25)

    # Print results
    print("Word-by-word evaluation:")
    for word, status in word_results:
        status_symbol = "✅" if status == "correct" else "❌"
        print(f"{status_symbol} {word} → {status}")

    # Summary of correctness
    correct_words = sum(1 for _, status in word_results if status == "correct")
    total_words = len(word_results)
    print(f"\nFinal Score: {correct_words}/{total_words} words read correctly.")
