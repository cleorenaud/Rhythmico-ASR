# This file contains code to ensure that the wav files are in the correct format for the model.

import os
import subprocess

# Folder containing your files
input_folder = "test_wav_files"
output_folder = "converted_wav_files"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Conversion settings
sample_rate = 16000  # Change if needed
channels = 1  # Mono audio

def convert_to_wav(input_file, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,            # input file
        '-ar', str(sample_rate),     # audio sample rate
        '-ac', str(channels),        # number of audio channels
        '-c:a', 'pcm_s16le',         # audio codec (PCM 16-bit little endian)
        output_file                  # output file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Converted: {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file}")
        print(e.stderr.decode())

# Loop through files in folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.wav', '.m4a')):
        input_path = os.path.join(input_folder, filename)

        # Output filename (strip extension and add _converted.wav)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}_converted.wav")

        # Convert the file
        convert_to_wav(input_path, output_path)