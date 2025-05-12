import argparse
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from TTS.api import TTS
from TTS.utils.manage import ModelManager
import os
import re
from tqdm import tqdm
import logging

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig

torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig
])

logging.basicConfig(level=logging.INFO)
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
    try:
        book = epub.read_epub(epub_path)
        chapters = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')

                for tag in soup(['script', 'style', 'nav']):
                    tag.decompose()

                chapter_title = get_chapter_title(soup)
                paragraphs = []

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


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', '', name.replace(' ', '_'))


def text_to_speech(text, output_file, model=None, language="en", speaker=None):
    # try:
        
        #tts = TTS(model_name="tts_models/de/thorsten/vits")
        #tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC")
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        
        tts.tts_to_file(
            text=text,
            file_path=output_file,
            language='de'
        )
        
        # if model is None:
        #     model = "tts_models/en/vctk/vits"

        # logger.info(f"Using TTS model: {model}")
        # logger.info(f"Output file: {output_file}")
        # logger.info(f"Text length: {len(text)} characters")
        # if language:
        #     logger.info(f"Language: {language}")

        # tts = TTS(model_name=model)

        # if hasattr(tts, 'speakers'):
        #     speakers = tts.speakers
        #     if speakers:
        #         if speaker is not None and 0 <= speaker < len(speakers):
        #             selected_speaker = speakers[speaker]
        #         else:
        #             selected_speaker = speakers[0]
        #         tts.tts_to_file(text=text, file_path=output_file, speaker=selected_speaker, language=language)
        # else:
        #     tts.tts_to_file(text=text, file_path=output_file, language=language)

    # except Exception as e:
    #     logger.error(f"Error in text-to-speech conversion: {e}")
    #     raise


def combine_audio_files(input_files, output_file):
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    for audio_file in input_files:
        audio = AudioSegment.from_wav(audio_file)
        combined += audio

    combined.export(output_file, format="mp3")


def main():
    parser = argparse.ArgumentParser(description='Convert epub to audio using TTS')
    parser.add_argument('epub_file', nargs='?', help='Path to the epub file')
    parser.add_argument('output_dir', nargs='?', help='Directory for audio output')
    parser.add_argument('--model', help='TTS model to use', default="tts_models/en/vctk/vits")
    parser.add_argument('--combine', action='store_true', help='Combine all chapters into single file')
    parser.add_argument('--language', help='Language code (e.g., en, de, fr)', default=None)
    parser.add_argument('--speaker', type=int, help='Speaker index for multi-speaker models', default=None)
    parser.add_argument('--list-speakers', action='store_true', help='List available speakers for the selected model')
    parser.add_argument('--list-models', action='store_true', help='List available TTS models')

    args = parser.parse_args()

    if args.list_models:
        try:
            manager = ModelManager()
            models = manager.list_models()
            print("\nAvailable TTS models:")
            for model in models:
                name = model.get("model_name", "unknown")
                lang = model.get("language", "n/a")
                speakers = "multi" if model.get("speaker") == "multi" else "single"
                print(f"{name}  |  Lang: {lang}  |  Speakers: {speakers}")
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return 1
        return 0

    if args.list_speakers:
        try:
            logger.info(f"Loading TTS model: {args.model}")
            tts = TTS(model_name=args.model)
            if hasattr(tts, 'speakers') and tts.speakers:
                print("\nAvailable speakers:")
                for idx, speaker in enumerate(tts.speakers):
                    print(f"{idx}: {speaker}")
            else:
                print("This model does not support multiple speakers.")
        except Exception as e:
            logger.error(f"Failed to load model or list speakers: {e}")
            return 1
        return 0

    if not args.epub_file or not args.output_dir:
        parser.error("epub_file and output_dir are required unless --list-speakers or --list-models is used")

    try:
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"Processing epub file: {args.epub_file}")
        chapters = epub_to_chapters(args.epub_file)

        chapter_files = []
        for i, chapter in enumerate(tqdm(chapters, desc="Converting chapters")):
            safe_title = sanitize_filename(chapter['title'][:30])
            chapter_file = os.path.join(args.output_dir, f"{i+1:03d}_{safe_title}.wav")
            logger.info(f"Converting chapter: {chapter['title']}")
            text_to_speech(chapter['content'], chapter_file, args.model, args.language, args.speaker)
            chapter_files.append(chapter_file)

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
