# Rhythmico Automatic Speech Recognition (ASR)

### Project Structure

As the `df_test_cleaned.csv` file is too big it is not stored on GitHub but it should be added as described in the following structure.
The wav files corresponding to each recording can be generated using the `save_recordings_as_wav(dataframe)` function in the `src/audio_processing.py` file.

```
├── data/
│   └── df_test_cleaned.csv                         # The test data
│ 
├── notebooks/
│   ├── evaluation                    
│   │   ├── readingTestFluencE_eval.ipynb           # Notebook to evaluate the readingTestFluencE
│   │   └── testPhoneme_eval.ipynb                  # Notebook to evaluate the testPhoneme
│   └── processing 
│       ├── readingTestFluencE_proc.ipynb           # Notebook to process the readingTestFluencE data
│       ├── rythmico.ipynb                          # Notebook for the initial data processing
│       └── testPhoneme_proc.ipynb                  # Notebook to process the testPhoneme data
│
├── src/                         
│   ├── audio_processing.py                         # Helper functions to process the tests audio recordings  
│   ├── data_processing.py                          # Helper functions to extract the tests data
│   ├── text_processing.py                          # Helper functions to save the phonetic transcription of a text to a .csv
│   └── ui_tools.py                                 # Helper functions to display audio players
│
├── transcriptions/
│   ├── readingTestFluencE_children.csv             # The phoneme transcriptions for each children recording for the readingTestFluencE
│   ├── readingTestFluencE_transcriptions.csv       # The ground truth for the readingTestFluencE
│   ├── testPhoneme_children.csv                    # The phoneme transcriptions for each children recording for the testPhoneme
│   ├── testPhoneme_deletion_transcriptions.csv     # The ground truth for the testPhoneme deletion
│   └── testPhoneme_fusion_transcriptions.csv       # The ground truth for the testPhoneme fusion
│
├── wav_files/
│   ├── readingTestFluencE                          # Folder containing the recordings for the readingTestFluencE
│   └── testPhoneme                                 # Folder containing the recordings for the testPhoneme
│
├── .gitignore
├── README.md
└── requirements.txt
```
