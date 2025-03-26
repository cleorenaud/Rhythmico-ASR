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