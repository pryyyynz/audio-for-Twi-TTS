#!/usr/bin/env python3
"""
Inference script for trained Coqui TTS Tacotron2 model on Twi language.
"""

import torch
import argparse
from TTS.api import TTS
from pathlib import Path
import soundfile as sf
import numpy as np


class TwiTTS:
    """Twi Text-to-Speech inference class"""

    def __init__(self, model_path=None, config_path=None, use_pretrained=True):
        """
        Initialize TTS model

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model config.json
            use_pretrained: Use pretrained multilingual model for quick start
        """
        self.model_path = model_path
        self.config_path = config_path
        self.tts = None

        if use_pretrained and not model_path:
            print("Loading pretrained multilingual XTTS model...")
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        elif model_path and config_path:
            print(f"Loading custom trained model from {model_path}")
            self.tts = TTS(model_path=model_path, config_path=config_path)
        else:
            raise ValueError(
                "Either use_pretrained=True or provide both model_path and config_path")

    def synthesize(self, text, output_path="output.wav", speaker_wav=None, language="tw"):
        """
        Synthesize speech from Twi text

        Args:
            text: Twi text to synthesize
            output_path: Output audio file path
            speaker_wav: Reference audio for voice cloning (optional)
            language: Language code (default: "tw" for Twi)
        """
        print(f"Synthesizing: '{text}'")

        try:
            if speaker_wav and Path(speaker_wav).exists():
                print(f"Using reference audio: {speaker_wav}")
                # Voice cloning with reference audio
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=speaker_wav,
                    language=language
                )
            else:
                # Standard TTS
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path
                )

            print(f"Audio saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error during synthesis: {e}")
            return None

    def batch_synthesize(self, texts, output_dir="outputs", speaker_wav=None):
        """
        Synthesize multiple texts

        Args:
            texts: List of Twi texts
            output_dir: Directory to save output files
            speaker_wav: Reference audio for voice cloning
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        results = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"synthesis_{i:03d}.wav"
            result = self.synthesize(text, str(output_path), speaker_wav)
            results.append(result)

        return results


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Twi Text-to-Speech Synthesis")
    parser.add_argument("--text", type=str, required=True,
                        help="Twi text to synthesize")
    parser.add_argument("--output", type=str,
                        default="output.wav", help="Output audio file")
    parser.add_argument("--model", type=str,
                        help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, help="Path to model config.json")
    parser.add_argument("--speaker", type=str,
                        help="Reference audio for voice cloning")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained multilingual model")

    args = parser.parse_args()

    # Initialize TTS
    tts_model = TwiTTS(
        model_path=args.model,
        config_path=args.config,
        use_pretrained=args.pretrained
    )

    # Synthesize
    result = tts_model.synthesize(
        text=args.text,
        output_path=args.output,
        speaker_wav=args.speaker
    )

    if result:
        print("Synthesis completed successfully!")
    else:
        print("Synthesis failed!")


def demo():
    """Demo function with sample Twi texts"""
    print("Running Twi TTS Demo...")

    # Sample Twi texts from your dataset
    sample_texts = [
        "Meda so ara wom.",
        "Ɛno na ɛboaa me.",
        "Dabi, ɛnte saa",
        "Yɛhyɛɛ no ma ogye toom sɛ ɔbɛyɛ.",
        "Woyɛ ɔbɔnefo."
    ]

    # Use a reference audio from your dataset
    reference_audio = "./clips/common_voice_tw_35280429.mp3"  # High quality sample

    # Initialize with pretrained model
    tts_model = TwiTTS(use_pretrained=True)

    # Create demo outputs
    demo_dir = Path("demo_outputs")
    demo_dir.mkdir(exist_ok=True)

    print(f"\nSynthesizing {len(sample_texts)} sample texts...")
    for i, text in enumerate(sample_texts):
        output_path = demo_dir / f"demo_{i+1:02d}.wav"
        print(f"\n{i+1}. {text}")

        if Path(reference_audio).exists():
            tts_model.synthesize(text, str(output_path),
                                 speaker_wav=reference_audio)
        else:
            tts_model.synthesize(text, str(output_path))

    print(f"\nDemo complete! Check outputs in {demo_dir}/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # Run demo if no arguments provided
        demo()
    else:
        # Run CLI
        main()
