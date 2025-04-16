# Rhythmico Automatic Speech Recognition (ASR)

### Project Structure

As the `df_test_cleaned.csv` file is too big it is not stored on GitHub but it should be added as described in the following structure.
The wav files corresponding to each recording can be generated using the `save_recordings_as_wav(dataframe)` function in the `src/audio_processing.py` file.

To run the `notebooks/evaluation/readingTestFluencE_Azure.ipynb`, one should define `AZURE_SPEECH_KEY` and `AZURE_REGION` in a .env file

The `wav2vec2-french-phonemizer` model can be found [here](https://huggingface.co/Cnam-LMSSC/wav2vec2-french-phonemizer)

```
├── data/
│   └── df_test_cleaned.csv                         # The test data
│ 
├── notebooks/
│   ├── evaluation
│   │   ├── readingTestFluencE_Azure.ipynb          # Notebook to process the readingTestFluencE using Azure              
│   │   ├── readingTestFluencE_eval.ipynb           # Notebook to evaluate the readingTestFluencE
│   │   ├── readingTestFluencE_top3.ipynb           # Notebook to evaluate the readingTestFluencE using the top 3 phoneme predictions
│   │   └── testPhoneme_eval.ipynb                  # Notebook to evaluate the testPhoneme
│   └── processing
│       ├── data_proc.ipynb                         # Notebook for the initial data processing
│       ├── readingTestFluencE_proc.ipynb           # Notebook to process the readingTestFluencE data
│       └── testPhoneme_proc.ipynb                  # Notebook to process the testPhoneme data
│
├── sample_readingTestFluencE/                      # Folder containing the data related to the 9 selected readingTestFluencE
│   ├── ...                                         # The recordings of the 9 selected readingTestFluencE
│   ├── readingTestFluencE_children.csv             # The manual phoneme transcriptions for 9 readingTestFluencE
│   └── sample_readingTestFluencE_text.csv          # The manual text transcriptions for 9 readingTestFluencE
│
├── src/                         
│   ├── audio_processing.py                         # Helper functions to process the tests audio recordings  
│   ├── data_processing.py                          # Helper functions to extract the tests data
│   ├── readingTestFluencE_eval_with_proba.py       # Custom evaluation functions for the readingTestFluencE using probabilities of phonemes
│   ├── readingTestFluencE_eval.py                  # Custom evaluation functions for the readingTestFluencE
│   ├── test_evaluation.py                          # Initial evaluation functions for the readingTestFluencE
│   ├── text_processing.py                          # Helper functions to save the phonetic transcription of a text to a .csv
│   ├── ui_tools.py                                 # Helper functions to display audio players
│   └── wav2vec2-french-phonemizer.py               # Functions to run the wav2vec2-french-phonemizer model 
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
