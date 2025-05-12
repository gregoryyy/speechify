# Speechify TTS system for epub files

Convert an epub ebook file to an audio book.

## Dependencies

- System (Mac):
  - brew install ffmpeg
  - brew install espeak-ng
- Python requirements.txt
  - Coqui TTS library to generate audio (formerly Mozilla TTS)
  - Beautiful Soup to parse epub content
  - PyTorch for ML
  - etc.

## Run

python speechify.py <input_file>.epub <output_dir>

## Models

Example models:
- **tts_models/multilingual/multi-dataset/xtts_v2**: Latest multilingual model supporting multiple languages with good quality.
- **tts_models/en/vctk/vits** (default): High-quality multi-speaker model with natural-sounding voices. Good for longer texts.
- **tts_models/de/thorsten/vits**: German model
- **tts_models/en/ljspeech/fast_pitch**: Optimized for speed while maintaining good quality. Single female voice.
- **tts_models/en/jenny/jenny**: Premium quality single female voice, excellent clarity and naturalness.
- **tts_models/de/thorsten/tacotron2-DDC**: German model

Coqui TTS models are downloaded automatically when first used, but they're stored locally at:

```
~/Library/Application Support/tts/tts_models/  # macOS
~/.local/share/tts/tts_models/  # Linux
%USERPROFILE%\.local\share\tts\tts_models\  # Windows
```
