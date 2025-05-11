import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from TTS.api import TTS
import os

def epub_to_text(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            chapters.append(soup.get_text())
    return chapters

def text_to_speech(text, output_file):
    tts = TTS.list_models()
    # Initialize TTS with a specific model
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    tts.tts_to_file(text=text, file_path=output_file)

# Usage
epub_path = "your_book.epub"
output_dir = "audio_output"
os.makedirs(output_dir, exist_ok=True)

chapters = epub_to_text(epub_path)
for i, chapter in enumerate(chapters):
    output_file = f"{output_dir}/chapter_{i}.wav"
    text_to_speech(chapter, output_file)

