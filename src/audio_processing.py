import io
import os
from pydub import AudioSegment
from IPython.display import Audio, display
import ipywidgets as widgets
import numpy as np
import torchaudio

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

def save_recordings_as_wav(dataframe, output_dir='converted_wav_files', target_sample_rate=16000, channels=1):
    """
    Given a dataframe, save all recordings as .wav files with the correct format (PCM 16-bit, mono, 16kHz).

    Parameters:
        dataframe (pd.DataFrame): Dataframe containing recordings in 'testResults' column.
        output_dir (str): Directory where converted .wav files will be saved.
        target_sample_rate (int): Desired sample rate for output files.
        channels (int): Number of channels for output files (1 for mono).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for _, row in dataframe.iterrows():
        # Extract the recording byte data from the 'testResults' column
        recording = row['testResults'].get('recording', None)

        if recording:
            try:
                # Get the test type, replace spaces with underscores to make filenames safe
                test_type = str(row.get('testType', 'unknown')).replace(' ', '_')

                # Construct the filename
                file_name = f"{test_type}_{row['id']}.wav"
                file_path = os.path.join(output_dir, file_name)

                # Load the recording bytes as an AudioSegment (assuming it's M4A)
                audio_segment = AudioSegment.from_file(io.BytesIO(recording), format="m4a")

                # Set the sample rate and channels
                audio_segment = audio_segment.set_frame_rate(target_sample_rate).set_channels(channels).set_sample_width(2)  # 2 bytes = 16 bits

                # Export to WAV with PCM 16-bit encoding
                audio_segment.export(file_path, format="wav")

            except Exception as e:
                print(f"Error processing recording for ID {row['id']}: {e}")

def load_audio(file_path):
    """
    Load a .wav file and return the waveform and sample rate.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def preprocess_audio(waveform, sample_rate=None, target_sample_rate=16000):
    """
    Preprocess the audio waveform.

    Since all audio files are already at the target sample rate, no resampling is performed.
    """
    if sample_rate != target_sample_rate:
        print(f"Warning: Expected sample rate {target_sample_rate}, but got {sample_rate}")
        # Optionally raise an error if you want strict checking:
        # raise ValueError(f"Sample rate mismatch: expected {target_sample_rate}, got {sample_rate}")
    return waveform
