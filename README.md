# glm4-audio-tokenizer

Standalone package GLM-4 Audio Tokenizer.

## Installation

```bash
pip3 install git+https://github.com/malaysia-ai/glm4-audio-tokenizer
```

## Encode

```python
from glm4_audio_tokenizer import Glm4Tokenizer

model = Glm4Tokenizer().cuda()
tokens = model.tokenize(['test.mp3'])
print(tokens)
```

Output,

```
[[12886,
  1698,
  10640,
  4353,
  2474,
  14101,
  5437,
  3734,
  8356,
  1462,
  1248,
  5545,
  3245,
  279,
  3752,
  5266,
  8074,
  15173,
  1188,
  1066,
  1066,
  6473,
  6473,
  7958,
  7958,
  7958,
  1066,
  7382,
  11363]]
```

### Tokenize

```
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
```