from transformers import WhisperFeatureExtractor
from .modeling_whisper import WhisperVQEncoder
from .utils import extract_speech_token
import torch
import librosa
import numpy as np
from torch import nn
from typing import List, Union, Tuple

class Glm4Tokenizer(nn.Module):
    def __init__(self, tokenizer_path = "THUDM/glm-4-voice-tokenizer"):
        super().__init__()
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    def tokenize(self, speech: List[Union[str, Tuple[np.ndarray, int]]], sample_rate = 16000):
        """
        Tokenizes a list of speech inputs into discrete tokens using the WhisperVQEncoder.

        Each item in `speech` can be one of the following:
        - A string: interpreted as a file path to a WAV audio file.
        - A tuple: (audio_array, sample_rate), where:
            - `audio_array` is a 1D or 2D numpy array (1D for mono, 2D for stereo).
            - `sample_rate` is an integer specifying the audio sample rate (in Hz).

        Args:
            speech (List[Union[str, Tuple[np.ndarray, int]]]):
                A list of either file paths to audio files or tuples of (audio tensor, sample rate).
        """
        audio = []
        for s in speech:
            if isinstance(s, str):
                y, sr = librosa.load(s, sr = sample_rate)
            elif isinstance(s, tuple) or isinstance(s, list):
                y, sr = s
                if y > 1:
                    y = y.mean(0)
                if sr != sample_rate:
                    y = librosa.resample(y, orig_sr = sr, target_sr = sample_rate)
            else:
                raise ValueError(f"Unsupported input type: {type(s)}")
            audio.append((y, sr))

        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, audio
        )
        return audio_tokens