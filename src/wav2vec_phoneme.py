import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer
from src.audio_processing import preprocess_audio

def load_model_and_tokenizer(model_name="Cnam-LMSSC/wav2vec2-french-phonemizer"):
    """
    Load the pre-trained model and tokenizer.
    """
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()  # Set model to evaluation mode
    return tokenizer, model

def predict_phonemes(waveform, tokenizer, model):
    """
    Predict phonemes for a given waveform.
    waveform: Tensor of shape [1, time]
    """
    # Preprocess the waveform to ensure sample rate = 16kHz
    waveform = preprocess_audio(waveform, target_sample_rate=16000)

    # Tokenize the waveform (expects numpy array)
    inputs = tokenizer(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)

    # Get the logits from the model
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits

    # Decode the predicted phonemes
    predicted_ids = torch.argmax(logits, dim=-1)
    phonemes = tokenizer.batch_decode(predicted_ids)[0]
    return phonemes
