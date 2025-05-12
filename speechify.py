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

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Allow model configs for PyTorch >= 2.6
torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs
])

# Logging to file to keep progress bar clean
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

def split_text_with_token_limit(text, tokenizer, max_tokens=400):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=max_tokens,
        chunk_overlap=0
    )
    return splitter.split_text(text)

def text_to_speech(text, output_file, model, language="de", speaker=None, speaker_wav=None):
    tts = TTS(model_name=model, gpu=True)

    if "xtts" in model:
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-de")
        chunks = split_text_with_token_limit(text, tokenizer)
    else:
        chunks = [text]  # No chunking required for non-XTTS models

    temp_files = []
    for i, chunk in enumerate(chunks):
        chunk_file = output_file.replace(".wav", f"_part{i+1}.wav")

        if "xtts" in model:
            if not speaker_wav:
                raise ValueError("XTTS requires --speaker-wav")
            tts.tts_to_file(text=chunk, file_path=chunk_file, language=language, speaker_wav=speaker_wav)
        elif hasattr(tts, "speakers") and tts.speakers:
            if speaker is None:
                selected_speaker = tts.speakers[0]
            elif isinstance(speaker, int):
                selected_speaker = tts.speakers[speaker]
            else:
                selected_speaker = speaker
            tts.tts_to_file(text=chunk, file_path=chunk_file, language=language, speaker=selected_speaker)
        else:
            tts.tts_to_file(text=chunk, file_path=chunk_file, language=language)

        temp_files.append(chunk_file)

    combined = AudioSegment.empty()
    for f in temp_files:
        combined += AudioSegment.from_wav(f)
    combined.export(output_file, format="wav")

    for f in temp_files:
        os.remove(f)

def main():
    parser = argparse.ArgumentParser(description="Convert EPUB to audio using Coqui TTS")
    parser.add_argument("epub_file", help="Path to the .epub file")
    parser.add_argument("output_dir", help="Directory for audio output")
    parser.add_argument("--model", default="tts_models/de/thorsten/vits", help="TTS model to use")
    parser.add_argument("--language", default="de", help="Language code (default: de)")
    parser.add_argument("--speaker", help="Speaker ID or index (for multi-speaker models)")
    parser.add_argument("--speaker-wav", help="Path to WAV file to use for XTTS zero-shot voice cloning")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Processing EPUB: {args.epub_file}")

    chapters = epub_to_chapters(args.epub_file)

    for i, chapter in enumerate(tqdm(chapters, desc="Converting chapters", unit="chapter")):
        safe_title = sanitize_filename(chapter["title"][:30])
        out_path = os.path.join(args.output_dir, f"{i+1:03d}_{safe_title}.wav")
        try:
            text_to_speech(
                text=chapter["content"],
                output_file=out_path,
                model=args.model,
                language=args.language,
                speaker=args.speaker,
                speaker_wav=args.speaker_wav
            )
        except Exception as e:
            logger.error(f"Failed to convert chapter '{chapter['title']}': {e}")

    print("✅ Conversion complete.")

if __name__ == "__main__":
    main()
