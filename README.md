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

. venv/bin/activate
python speechify.py <options> <input_file.epub> <output_dir>

## Models

Example models:
- **tts_models/multilingual/multi-dataset/xtts_v2**: Latest multilingual model supporting multiple languages with good quality. Supports single-shot speech cloning.
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


## Speech samples

- For XTTS zero-shot voice cloning
  - prereq: brew install yt-dlp
  - youtube download audio: yt-dlp -x --audio-format wav -o <outfile> <url>
  - trim audio: ffmpeg -i <infile> -ss <start timestamp> -t <length> -ac 1 -ar 16000 -y <outfile>
  - 3-10 seconds suffice
```
# American English:
# Obama: 
yt-dlp -x --audio-format wav -o "obama.wav" "https://www.youtube.com/watch?v=nU3E8r0n27w"
ffmpeg -i obama.wav -ss 00:21:26 -t 00:00:13 -ac 1 -ar 16000 -y obama_sample1.wav
# Kennedy:
yt-dlp -x --audio-format wav -o "kennedy.wav" "https://www.youtube.com/watch?v=RclaV_3_eOA"
ffmpeg -i kennedy.wav -ss 00:00:20.500 -t 00:00:14 -ac 1 -ar 16000 -y kennedy_sample1.wav
# German:
# Burghart Klaußner:
yt-dlp -x --audio-format wav -o "klaussner.wav" "https://www.youtube.com/watch?v=LzFTsOd-xMg"
ffmpeg -i klaussner.wav -ss 00:00:33.000 -t 00:00:14 -ac 1 -ar 16000 -y klaussner_sample1.wav
# OMR Guests:
# Saskia Meyer-Andrae, Ebay.de
yt-dlp -x --audio-format wav -o "meyer.wav" "https://www.youtube.com/watch?v=th3pDwTagdQ"
ffmpeg -i meyer.wav -ss 00:08:17.000 -t 00:00:14 -ac 1 -ar 16000 -y meyer_sample1.wav
# Josef Ackermann (Swiss accent):
yt-dlp -x --audio-format wav -o "ackermann.wav" "https://www.youtube.com/watch?v=xyNodncJ_2c"
ffmpeg -i ackermann.wav -ss 00:11:35.000 -t 00:00:14 -ac 1 -ar 16000 -y ackermann_sample1.wav
# Philipp Westermeyer:
ffmpeg -i ackermann.wav -ss 00:04:05.500 -t 00:00:14 -ac 1 -ar 16000 -y westermeyer_sample1.wav
```