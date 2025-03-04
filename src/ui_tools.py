from IPython.display import display

def display_words_state(words_dict):
    """Displays the word states in a widget."""
    text = "\n".join(f"{word}: {'✅' if correct else '❌'}" for word, correct in words_dict.items())
    display(text)

def update_audio_and_results(audio_func, data_func, index, data):
    """Updates the displayed audio and results."""
    audio = audio_func(data[index]['audio'])
    results = data_func(data[index]['words'])
    display(audio)
    display_words_state(results)
    
def on_prev_clicked(change, index, data, update_func):
    """Handles previous button click event."""
    if index > 0:
        index -= 1
        update_func(index, data)
    return index

def on_next_clicked(change, index, data, update_func):
    """Handles next button click event."""
    if index < len(data) - 1:
        index += 1
        update_func(index, data)
    return index
