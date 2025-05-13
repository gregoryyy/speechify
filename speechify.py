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

import spacy
try:
    nlp = spacy.load("xx_sent_ud_sm")
except OSError:
    print("Model not installed. Run: python -m spacy download xx_sent_ud_sm")
    raise

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
    '''Clean and normalize text by removing extra spaces and newlines.'''
    return ' '.join(text.split()).strip()

def get_chapter_title(soup):
    '''Extract the title of the chapter from the HTML content.'''
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
        title = clean_text(tag.get_text())
        if title:
            return title
    return None

def epub_to_chapters(epub_path):
    '''Convert EPUB file to chapters with text content.'''
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
    '''Sanitize filename by removing invalid characters.'''
    return re.sub(r'[\\/*?:"<>|]', '', name.replace(' ', '_'))


import spacy
import re

nlp = spacy.load("en_core_web_sm")  # Replace with your language model if needed

def preprocess_text(text, max_chars=250):
    """Split text into sentence-aware chunks that do not exceed max_chars each."""
    doc = nlp(text)
    chunks = []
    current_chunk = ""

    def strip_special_chars(sentence):
        return re.sub(r"[^\w\s.,!?]", "", sentence)

    for sent in doc.sents:
        sentence = strip_special_chars(sent.text.strip())

        if len(sentence) > max_chars:
            # Split overly long sentence into smaller pieces at word level
            words = sentence.split()
            part = ""
            for word in words:
                if len(part) + len(word) + 1 <= max_chars:
                    part = f"{part} {word}".strip()
                else:
                    if part:
                        chunks.append(part)
                    part = word
            if part:
                chunks.append(part)
            continue

        # Try adding the sentence to the current chunk
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def get_device():
    '''Get the appropriate accelerator for TTS and ML processing.'''
    if torch.cuda.is_available():
        logger.info("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.info("Using MPS")
        return torch.device("mps")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")

def text_to_speech(text, output_file, speaker_wav, language="de", output_format="wav", accelerate=False):
    '''Convert text to speech using XTTS v2.'''
    if os.path.exists(output_file):
        logger.info(f"Skipping existing file: {output_file}")
        return

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    if accelerate == "true":
        device = get_device()
        tts.to(device)
    
    # XTTS is max 253 chars for German.
    chunks = preprocess_text(text, max_chars=250)

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
    '''Combine multiple audio files into a single file.'''
    combined = AudioSegment.empty()
    for file_path in file_list:
        combined += AudioSegment.from_file(file_path)
    combined.export(output_file, format=output_file.split(".")[-1])

def main():
    parser = argparse.ArgumentParser(description="Convert EPUB to audio using XTTS v2")
    parser.add_argument("input_file", help="Path to the .epub or .txt file")
    parser.add_argument("output_dir", nargs="?", default=None, help="Directory for audio output")
    parser.add_argument("--language", default="de", help="Language code (default: de)")
    parser.add_argument("--speaker-wav", help="Path to WAV file to use for XTTS zero-shot voice cloning")
    parser.add_argument("--format", default="wav", choices=["wav", "mp3"], help="Output audio format")
    parser.add_argument("--combine", action="store_true", help="Combine all chapter files into a single audiobook")
    parser.add_argument("--accelerate", default="false", help="Enable GPU or MPS acceleration")
    parser.add_argument("--text-only", action="store_true", help="Only output parsed chapter text, no audio")
    args = parser.parse_args()

    logger.info(f"Processing file: {args.input_file}")
    if args.input_file.endswith(".txt"):
        chapters = []
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            paragraphs = [clean_text(p) for p in content.split('\n===') if clean_text(p)]
            for i, paragraph in enumerate(paragraphs):
                chapters.append({
                    'title': f"Chapter {i + 1}",
                    'content': paragraph
                })
    elif args.input_file.endswith(".epub"):
        chapters = epub_to_chapters(args.input_file)
    else:
        parser.error("Unsupported file format. Please provide a .epub or .txt file.")

    if args.text_only:
        for i, chapter in enumerate(chapters):
            print(f"\n=== Chapter {i+1}: {chapter['title']} ===\n")
            print(chapter["content"])
        return

    if not args.output_dir:
        parser.error("--output_dir is required unless --text-only is set.")

    if not args.speaker_wav:
        parser.error("--speaker-wav is required unless --text-only is set.")

    os.makedirs(args.output_dir, exist_ok=True)

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
                output_format=args.format,
                accelerate=args.accelerate == "true"
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
