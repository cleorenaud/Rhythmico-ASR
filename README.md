# Rhythmico Automatic Speech Recognition (ASR)

### Project Structure

As the `df_test_cleaned.csv` file is too big it is not stored on GitHub but it should be added as described in the following structure.
The wav files corresponding to each recording can be generated using the `save_recordings_as_wav(dataframe)` function in the `src/audio_processing.py` file.

To run the `notebooks/evaluation/readingTestFluencE_Azure.ipynb`, one should define `AZURE_SPEECH_KEY` and `AZURE_REGION` in a .env file

The three following models are used and compared in this project:
- The `Cnam-LMSSC/wav2vec2-french-phonemizer` model can be found [here](https://huggingface.co/Cnam-LMSSC/wav2vec2-french-phonemizer)
- The `bofenghuang/phonemizer-wav2vec2-ctc-french` model can be found [here](https://huggingface.co/bofenghuang/phonemizer-wav2vec2-ctc-french/tree/main)
- The `Bluecast/wav2vec2-Phoneme` model can be found [here](https://huggingface.co/Bluecast/wav2vec2-Phoneme)

```
├── data/
│   └── df_test_cleaned.csv                         # The test data
│ 
├── notebooks/
│   ├── evaluation
│   │   ├── readingTestFluencE_Azure.ipynb          # Notebook to process the readingTestFluencE using Azure              
│   │   ├── readingTestFluencE_eval.ipynb           # Notebook to evaluate the readingTestFluencE
│   │   ├── readingTestFluencE_top3_eval.ipynb      # Notebook to evaluate the readingTestFluencE using the top 3 phoneme predictions
│   │   ├── readingTestNonWords_top3_eval.ipynb     # Notebook to evaluate the readingTestNonWords using the top 3 phoneme predictions
│   │   └── testPhoneme_eval.ipynb                  # Notebook to evaluate the testPhoneme
│   ├── processing
│   │   ├── data_proc.ipynb                         # Notebook for the initial data processing
│   │   ├── readingTestFluencE_proc.ipynb           # Notebook to process the readingTestFluencE data
│   │   ├── readingTestNonWords_proc.ipynb          # Notebook to process the readingTestNonWords data
│   │   └── testPhoneme_proc.ipynb                  # Notebook to process the testPhoneme data
│   └── statistics.ipynb                            # Notebook to extract statistics about the phoneme repartitions and the performance of models
│
├── src/                         
│   ├── audio_processing.py                         # Helper functions to process the tests audio recordings  
│   ├── data_processing.py                          # Helper functions to extract the tests data
│   ├── phoneme_stats.py                            # Helper functions to extract statistics about the phonemes
│   ├── readingTestFluencE_eval_with_proba.py       # Custom evaluation functions for the readingTestFluencE using probabilities of phonemes
│   ├── readingTestFluencE_eval.py                  # Custom evaluation functions for the readingTestFluencE
│   ├── readingTestNonWords_eval.py                 # Custom evaluation functions for the readingTestNonWords using probabilities of phonemes
│   ├── test_evaluation.py                          # Initial evaluation functions for the readingTestFluencE
│   ├── text_processing.py                          # Helper functions to save the phonetic transcription of a text to a .csv
│   ├── ui_tools.py                                 # Helper functions to display audio players
│   └── wav2vec2_models.py                          # Functions to run the different wav2vec2 models
│
├── transcriptions/
│   └── readingTestFluencE/                        
│       ├── phonemizer-wav2vec2-ctc-french/        # Folder containing the transcription in phonemes made by the phonemizer-wav2vec2-ctc-french model for readingTestFluencE
│       ├── wav2vec2-french-phonemizer/            # Folder containing the transcription in phonemes made by the wav2vec2-french-phonemizer model for readingTestFluencE
│       └── wav2vec2-Phoneme/                      # Folder containing the transcription in phonemes made by the wav2vec2-Phoneme model for readingTestFluencE
│
├── wav_files/
│   ├── readingTestFluencE/                        # Folder containing the recordings for the readingTestFluencE
│   ├── readingTestNonWords/                       # Folder containing the recordings for the readingTestNonWords           
│   └── testPhoneme/                               # Folder containing the recordings for the testPhoneme
│
├── .gitignore
├── README.md
└── requirements.txt
```
