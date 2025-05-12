import torch
from TTS.api import TTS
import os

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

# Allow model configs for PyTorch >= 2.6
torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs
])

## Note: On M1 MacBook Pro, this leads to:
## NotImplementedError: Output channels > 65536 not supported at the MPS device. 

def get_best_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        print("✅ Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("✅ Using MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("✅ Using CPU")
        return torch.device("cpu")

def synthesize_text(text, output_path, speaker_wav=None, language="en"):
    """Synthesize text using Coqui XTTS v2."""
    device = get_best_device()

    # Initialize TTS
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    tts.to(device)

    # XTTS v2 requires a speaker WAV file for zero-shot voice cloning
    if speaker_wav is None:
        raise ValueError("XTTS v2 requires a speaker WAV file. Provide via --speaker-wav.")

    # Synthesize and save
    print(f"🎤 Synthesizing: {text}")
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=output_path)
    print(f"📁 Saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    example_text = "Hello! This is a test of adaptive device selection with XTTS v2."
    output_file = "output.wav"
    speaker_wav_path = "../voices/kennedy_sample1.wav"

    if not os.path.exists(speaker_wav_path):
        print("❌ Please provide a valid speaker WAV file at:", speaker_wav_path)
    else:
        synthesize_text(example_text, output_file, speaker_wav=speaker_wav_path, language="en")
