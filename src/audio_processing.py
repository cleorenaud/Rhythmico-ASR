import io
import os
from pydub import AudioSegment
from IPython.display import Audio, display
import ipywidgets as widgets
import numpy as np
import pandas as pd
import base64

def create_audio_player_with_results(recordings, evaluation_results):
    """
    Creates an audio player with Previous/Next buttons and displays the words expected and the evaluation results

    Parameters:
        recordings (list): List of audio data
        evaluation_results (list): List of evaluation results corresponding to each recording
    """

    if not recordings or not evaluation_results:
        print("No recordings or evaluation results available.")
        return
    
    # Create an index tracker
    index = widgets.IntText(value=0, min=0, max=len(recordings)-1, description="Index:")

    # Create buttons to navigate
    prev_button = widgets.Button(description="Previous")
    next_button = widgets.Button(description="Next")
    audio_output = widgets.Output()
    evaluation_output = widgets.Output()

    # Function to display the word states for the current evaluation
    def display_words_state(words_state):
        """Displays word states in a readable format."""

        with evaluation_output:
            evaluation_output.clear_output()
            for entry in words_state:
                for word, state in entry.items():
                    print(f"{word}: {state.lower()}")

    # Callback functions for buttons
    def update_audio_and_results(_=None):
        """Updates the audio output and evaluation results only when a button is pressed."""
        
        audio_output.clear_output()
        with audio_output:
            display(play_m4a(recordings[index.value]))

        # Display corresponding evaluation results
        display_words_state(evaluation_results[index.value])

    def on_prev_clicked(b):
        if index.value > 0:
            index.value -= 1
            update_audio_and_results()

    def on_next_clicked(b):
        if index.value < len(recordings) - 1:
            index.value += 1
            update_audio_and_results()

    # Link buttons to functions
    prev_button.on_click(on_prev_clicked)
    next_button.on_click(on_next_clicked)

    # Display the first audio player and its evaluation result
    update_audio_and_results()

    # Arrange widgets
    display(widgets.HBox([prev_button, index, next_button]), audio_output, evaluation_output)

def play_m4a(encoded_m4a):     
    """Plays an M4A audio file from encoded data."""

    m4a_buffer = io.BytesIO(encoded_m4a)        
    audio = AudioSegment.from_file(m4a_buffer, format="m4a")   
    audio_np = np.array(audio.get_array_of_samples())

    # If the audio is stereo, reshape the numpy array
    if audio.channels == 2:
        audio_np = audio_np.reshape((-1, 2))

    return Audio(audio_np, rate=audio.frame_rate)

def save_as_mp3(encoded_m4a, filename, output_folder="mp3_files"):
    """Converts and saves M4A audio as MP3 in the specified folder."""
    
    os.makedirs(output_folder, exist_ok=True)
    m4a_buffer = io.BytesIO(encoded_m4a)
    audio = AudioSegment.from_file(m4a_buffer, format="m4a")
    mp3_path = os.path.join(output_folder, f"{filename}.mp3")
    audio.export(mp3_path, format="mp3")
    return mp3_path

def save_recordings_as_wav(dataframe, output_dir='wav_files'):
    """Given a dataframe, save all recordings as .wav files to the specified directory."""
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate through each row in the DataFrame
    for index, row in dataframe.iterrows():
        # Extract the recording byte data from the 'testResults' column
        recording = row['testResults'].get('recording', None)
        
        if recording:
            try:
                # Construct the filename
                file_name = f"recording_{row['id']}.wav"
                file_path = os.path.join(output_dir, file_name)
                
                # Save the byte data as a .wav file
                with open(file_path, 'wb') as f:
                    f.write(recording)
                
                print(f"Saved {file_name}")
            
            except Exception as e:
                print(f"Error processing recording for ID {row['id']}: {e}")

