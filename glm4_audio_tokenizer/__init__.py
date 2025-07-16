import torch
import librosa
import os

from transformers import WhisperFeatureExtractor
from .modeling_whisper import WhisperVQEncoder
from .utils import extract_speech_token
from torch import nn

class Glm4Tokenizer(nn.Module):
    def __init__(self, tokenizer_path = "THUDM/glm-4-voice-tokenizer"):
        super().__init__()
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    def tokenize(self, speech=None, audio_path=None, sr=16000):
        if audio_path:
            audio, sr = librosa.load(audio_path, sr=16000)
            audio_info = (audio, sr)
        else:
            assert speech is not None
            assert sr
            if isinstance(speech, list):
                speech = torch.tensor(speech)
            audio_info = (speech, sr)

        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [audio_info]
        )
        return audio_tokens