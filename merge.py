import os
import argparse
import subprocess
from pydub import AudioSegment
import re
from tqdm import tqdm

def adjust_speed_ffmpeg(input_file, output_file, speed):
    if speed == 1.0:
        return
    temp_file = output_file.replace(".", "_temp.")
    os.rename(output_file, temp_file)

    if not 0.5 <= speed <= 2.0:
        raise ValueError("Speed must be between 0.5 and 2.0 for pitch-preserving mode.")

    cmd = [
        "ffmpeg", "-y", "-i", temp_file,
        "-filter:a", f"atempo={speed}",
        output_file
    ]
    subprocess.run(cmd, check=True)
    os.remove(temp_file)

def merge_audio_files(directory, output_file, speed=1.0):
    files = sorted(
        [f for f in os.listdir(directory) if f.lower().endswith(('.wav', '.mp3'))],
        key=lambda x: int(re.match(r'(\d+)', x).group(1)) if re.match(r'(\d+)', x) else float('inf')
    )

    if not files:
        print("No audio files found in the directory.")
        return

    combined = AudioSegment.empty()
    for file in tqdm(files, desc="Merging audio files", unit="file"):
        audio = AudioSegment.from_file(os.path.join(directory, file))
        combined += audio

    combined.export(output_file, format=output_file.split('.')[-1])
    adjust_speed_ffmpeg(output_file, output_file, speed)
    print(f"✅ Merged {len(files)} files into {output_file} at {speed}x speed")

def main():
    parser = argparse.ArgumentParser(description="Merge audio files output by speechify into one file")
    parser.add_argument("input_dir", help="Directory containing audio files (e.g., output from speechify.py)")
    parser.add_argument("output_file", help="Path to final merged audio file (e.g., merged.mp3 or merged.wav)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g., 1.0 = normal, 1.2 = 20% faster, pitch preserved)")
    args = parser.parse_args()

    merge_audio_files(args.input_dir, args.output_file, speed=args.speed)

if __name__ == "__main__":
    main()