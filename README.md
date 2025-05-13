# Speechify TTS system for epub files

Convert an epub ebook file to an audio book.

Author: Gregor Heinrich gregor :: arbylon . net
Date:   20250511

## Dependencies

- System (Mac):
  - brew install ffmpeg
  - brew install espeak-ng
- Python requirements.txt
  - Coqui TTS library to generate audio (formerly Mozilla TTS)
  - Beautiful Soup to parse epub content
  - PyTorch for ML
  - etc.

## Setup

Prerequisite: 

1 -- Open and setup environment:

```
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

2 -- Prepare data:

- epub files to read
- voice samples as wav files whose voice will be used (3-15 seconds low noise, phonetic variation, works, cross-language)


## Run

### Speechify

```
python speechify.py <options> <input_file.epub> <output_dir>
```

Synopsis:

```
Convert EPUB to audio using XTTS v2

positional arguments:
  input_file            Path to the .epub or .txt file
  output_dir            Directory for audio output

options:
  -h, --help            show this help message and exit
  --language LANGUAGE   Language code (default: de)
  --speaker-wav SPEAKER_WAV
                        Path to WAV file to use for XTTS zero-shot voice cloning
  --format {wav,mp3}    Output audio format
  --combine             Combine all chapter files into a single audiobook (default: false)
  --accelerate true/false
                        Enable GPU or MPS acceleration (default: false)
  --text-only           Only output parsed chapter text, no audio
```

Example:

```
python speechify.py --speaker-wav kennedy.wav --language en --format mp3 book.epub book_dir
```

### Merge

If audio files need to be merged and speed-adjusted (resampled):

```
python merge.py [--speed <0.5 to 2.0>] <input_dir> <output_file>
```

### Test TTS

To check whether ``speechify --accelerate`` works on the machine:

```
python test_tts.py
```


## Models

Note that the required parameters differ per model

Example models:
- **tts_models/multilingual/multi-dataset/xtts_v2**: Latest multilingual model supporting multiple languages with good quality.
  - Supports zero-shot voice cloning.
  - ``--speaker-wav <file with voice sample>``
  - Token-based text splitting

Note: XTTS is the only current option. Other Coqui models would need extension in speechify, e.g.:
- **tts_models/en/vctk/vits** (default): High-quality multi-speaker model with natural-sounding voices. Good for longer texts.
  - Multi-speaker model.
  - ``--speaker <speaker ID or index>``
- **tts_models/en/ljspeech/fast_pitch**: Optimized for speed while maintaining good quality. Single female voice.
- **tts_models/en/jenny/jenny**: Premium quality single female voice, excellent clarity and naturalness.
- **tts_models/de/thorsten/vits**: German model
- **tts_models/de/thorsten/tacotron2-DDC**: German model

Coqui TTS models are downloaded automatically when first used, but they're stored locally at:

```
~/Library/Application Support/tts/tts_models/  # macOS
~/.local/share/tts/tts_models/  # Linux
%USERPROFILE%\.local\share\tts\tts_models\  # Windows
```


## Speaker samples

For XTTS zero-shot voice cloning:
- prerequisite: brew install yt-dlp
- youtube download audio: yt-dlp -x --audio-format wav -o <outfile> <url>
- trim audio: ffmpeg -i <infile> -ss <start timestamp> -t <length> -ac 1 -ar 16000 -y <outfile>
- 3-10 seconds suffice
- Note: Usage is likely subject to license agreements beyond private and fair use

```
# American English:
# Obama: 
yt-dlp -x --audio-format wav -o "obama.wav" "https://www.youtube.com/watch?v=nU3E8r0n27w"
ffmpeg -i obama.wav -ss 00:21:26 -t 00:00:13 -ac 1 -ar 16000 -y obama_sample1.wav
# Kennedy:
yt-dlp -x --audio-format wav -o "kennedy.wav" "https://www.youtube.com/watch?v=RclaV_3_eOA"
ffmpeg -i kennedy.wav -ss 00:00:20.500 -t 00:00:14 -ac 1 -ar 16000 -y kennedy_sample1.wav
# Jon Hamm:
yt-dlp -x --audio-format wav -o "hamm.wav" "https://www.youtube.com/watch?v=RiHnqB66VN4"
ffmpeg -i hamm.wav -ss 00:01:39.500 -t 00:00:14 -ac 1 -ar 16000 -y hamm_sample1.wav
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
# Beate
yt-dlp -x --audio-format wav -o "beate.wav" "https://www.youtube.com/watch?v=vFbYEBmuwcc"
ffmpeg -i beate.wav -ss 00:01:52.000 -t 00:00:14 -ac 1 -ar 16000 -y beate_sample1.wav


```