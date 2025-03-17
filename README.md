# Rhythmico Automatic Speech Recognition (ASR)

### Project Structure

As the `df_test_cleaned.csv` file is too big it is not stored on GitHub but it should be added as described in the following structure.
The wav files corresponding to each recording can be generated using the `save_recordings_as_wav(dataframe)` function in the `src/audio_processing.py` file.

```
├── data
│   ├── df_test_cleaned.csv             # The test data
│ 
├── src                         
│   ├── audio_processing.py             # Helper functions to process the tests audio recordings  
│   ├── convert_to_wav.py               # Helper function to ensure the files are .wav and convert them otherwise
│   ├── data_processing.py              # Helper functions to extract the tests data
│   ├── text_processing.py              # Helper function to save the phonetic transcription of a text to a .csv
│   ├── ui_tools.py                     # Helper functions to display audio players
│
├── transcriptions                      # Folder containing .csv files with phonetic transcriptions
│
├── wav_files                           # Folder containing the recordings of each test
│
├── .gitignore
├── README.md
├── requirements.txt
├── rythmico.ipynb                      # Notebook to process the test data
├── text-to-phoneme.ipynb               # Notebook to translate a text to its corresponding phonemes
└── wav2vec2-french-phonemizer.ipynb    # Notebook to run the wav2vec2-french-phonemizer model on the test recordings
```
