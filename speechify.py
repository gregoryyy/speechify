import argparse
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from TTS.api import TTS
import os
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean and normalize text."""
    return ' '.join(text.split()).strip()

def get_chapter_title(soup):
    """Extract chapter title from HTML."""
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
        title = clean_text(tag.get_text())
        if title:
            return title
    return None

def epub_to_chapters(epub_path):
    """Convert epub to structured chapters with titles and content."""
    try:
        book = epub.read_epub(epub_path)
        chapters = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                
                # Remove unwanted elements
                for tag in soup(['script', 'style', 'nav']):
                    tag.decompose()
                
                chapter_title = get_chapter_title(soup)
                paragraphs = []
                
                # Process content by paragraph
                for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    text = clean_text(p.get_text())
                    if text:
                        paragraphs.append(text)
                
                if paragraphs:
                    chapters.append({
                        'title': chapter_title or f"Chapter {len(chapters) + 1}",
                        'content': '\n'.join(paragraphs)
                    })
        
        return chapters
    
    except Exception as e:
        logger.error(f"Error processing epub file: {e}")
        raise

def text_to_speech(text, output_file, model=None):
    """Convert text to speech using specified TTS model."""
    try:
        if model is None:
            # Default model selection
            model = "tts_models/en/vctk/vits"  # Multi-speaker model
            # Alternative options:
            # model = "tts_models/en/ljspeech/fast_pitch"  # Fast single female voice
            # model = "tts_models/en/jenny/jenny"  # High quality female voice
            # model = "tts_models/multilingual/multi-dataset/xtts_v2"  # Multilingual
        
        tts = TTS(model_name=model)
        tts.tts_to_file(text=text, file_path=output_file)
        
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {e}")
        raise

def combine_audio_files(input_files, output_file):
    """Combine multiple audio files into one."""
    from pydub import AudioSegment
    
    combined = AudioSegment.empty()
    for audio_file in input_files:
        audio = AudioSegment.from_wav(audio_file)
        combined += audio
    
    combined.export(output_file, format="mp3")

def main():
    parser = argparse.ArgumentParser(description='Convert epub to audio using TTS')
    parser.add_argument('epub_file', help='Path to the epub file')
    parser.add_argument('output_dir', help='Directory for audio output')
    parser.add_argument('--model', help='TTS model to use', default=None)
    parser.add_argument('--combine', action='store_true', help='Combine all chapters into single file')
    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process epub
        logger.info(f"Processing epub file: {args.epub_file}")
        chapters = epub_to_chapters(args.epub_file)
        
        chapter_files = []
        # Convert each chapter to audio
        for i, chapter in enumerate(tqdm(chapters, desc="Converting chapters")):
            chapter_file = os.path.join(
                args.output_dir,
                f"{i+1:03d}_{chapter['title'][:30]}.wav".replace(' ', '_')
            )
            
            logger.info(f"Converting chapter: {chapter['title']}")
            text_to_speech(chapter['content'], chapter_file, args.model)
            chapter_files.append(chapter_file)
            
        # Combine audio files if requested
        if args.combine and chapter_files:
            output_file = os.path.join(args.output_dir, "combined_audiobook.mp3")
            logger.info("Combining audio files...")
            combine_audio_files(chapter_files, output_file)
            
        logger.info("Conversion completed successfully!")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())