import argparse
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from TTS.api import TTS
import os
import re
from tqdm import tqdm
import logging
from pydub import AudioSegment


import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab') 

from nltk.tokenize import sent_tokenize
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

# Logging
logging.basicConfig(
    filename="speechify.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def clean_text(text):
    return ' '.join(text.split()).strip()

def get_chapter_title(soup):
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
        title = clean_text(tag.get_text())
        if title:
            return title
    return None

def epub_to_chapters(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            for tag in soup(['script', 'style', 'nav']):
                tag.decompose()
            title = get_chapter_title(soup)
            paragraphs = [clean_text(p.get_text()) for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5']) if clean_text(p.get_text())]
            if paragraphs:
                chapters.append({
                    'title': title or f"Chapter {len(chapters) + 1}",
                    'content': '\n'.join(paragraphs)
                })
    return chapters

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', '', name.replace(' ', '_'))

def split_text_nltk(text, max_chars=400):
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks

def text_to_speech(text, output_file, speaker_wav, language="de", output_format="wav"):
    if os.path.exists(output_file):
        logger.info(f"Skipping existing file: {output_file}")
        return

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
    chunks = split_text_nltk(text, max_chars=500)

    temp_files = []
    for i, chunk in enumerate(chunks):
        chunk_file = output_file.replace(f".{output_format}", f"_part{i+1}.wav")
        if not speaker_wav:
            raise ValueError("XTTS requires --speaker-wav")
        tts.tts_to_file(text=chunk, file_path=chunk_file, language=language, speaker_wav=speaker_wav)
        temp_files.append(chunk_file)

    combined = AudioSegment.empty()
    for f in temp_files:
        combined += AudioSegment.from_wav(f)
    combined.export(output_file, format=output_format)

    for f in temp_files:
        os.remove(f)

def combine_chapter_files(file_list, output_file):
    combined = AudioSegment.empty()
    for file_path in file_list:
        combined += AudioSegment.from_file(file_path)
    combined.export(output_file, format=output_file.split(".")[-1])

def main():
    parser = argparse.ArgumentParser(description="Convert EPUB to audio using XTTS v2")
    parser.add_argument("epub_file", help="Path to the .epub file")
    parser.add_argument("output_dir", help="Directory for audio output")
    parser.add_argument("--language", default="de", help="Language code (default: de)")
    parser.add_argument("--speaker-wav", required=True, help="Path to WAV file to use for XTTS zero-shot voice cloning")
    parser.add_argument("--format", default="wav", choices=["wav", "mp3"], help="Output audio format")
    parser.add_argument("--combine", action="store_true", help="Combine all chapter files into a single audiobook")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Processing EPUB: {args.epub_file}")

    chapters = epub_to_chapters(args.epub_file)

    output_files = []
    for i, chapter in enumerate(tqdm(chapters, desc="Converting chapters", unit="chapter")):
        safe_title = sanitize_filename(chapter["title"][:30])
        out_path = os.path.join(args.output_dir, f"{i+1:03d}_{safe_title}.{args.format}")
        try:
            text_to_speech(
                text=chapter["content"],
                output_file=out_path,
                speaker_wav=args.speaker_wav,
                language=args.language,
                output_format=args.format
            )
            output_files.append(out_path)
        except Exception as e:
            logger.error(f"Failed to convert chapter '{chapter['title']}': {e}")

    if args.combine and output_files:
        combined_path = os.path.join(args.output_dir, f"combined_audiobook.{args.format}")
        print("🔗 Combining chapter files into one audiobook...")
        combine_chapter_files(output_files, combined_path)

    print("✅ Conversion complete.")

if __name__ == "__main__":
    main()
