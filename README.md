# Rhythmico Automatic Speech Recognition (ASR)

### Project Structure

As the `df_test_cleaned.csv` file is too big it is not stored on GitHub but it should be added as described in the following structure.
The wav files corresponding to each recording can be generated using the `save_recordings_as_wav(dataframe)` function in the `src/audio_processing.py` file.

```
├── data
│   ├── df_test_cleaned.csv
│ 
├── src                         
│   ├── audio_processing.py     
│   ├── data_processing.py
│   ├── ui_tools.py
│
├── wav_files
│
├── rythmico.ipynb
├── .gitignore
├── pip_requirements.txt
└── README.md
```
