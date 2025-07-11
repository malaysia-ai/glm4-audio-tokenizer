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
tokens = model.tokenize(audio_path = 'test.mp3')
print(tokens, tokens.shape)
```

Output,

```
tensor([[12886,  1698, 10640,  4353,  2474, 14101,  5437,  3734,  8356,  1462,
          1248,  5545,  3245,   279,  3752,  5266,  8074, 15173,  1188,  1066,
          1066,  6473,  6473,  7958,  7958,  7958,  1066,  7382, 11363]]) torch.Size([1, 29])
```