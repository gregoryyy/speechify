import os
import argparse
from pydub import AudioSegment
import re
from tqdm import tqdm

def merge_audio_files(directory, output_file):
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
    print(f"✅ Merged {len(files)} files into {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge audio files output by speechify into one file")
    parser.add_argument("input_dir", help="Directory containing audio files (e.g., output from speechify.py)")
    parser.add_argument("output_file", help="Path to final merged audio file (e.g., merged.mp3 or merged.wav)")
    args = parser.parse_args()

    merge_audio_files(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()