#!/usr/bin/env python3
"""
Alternative inference script for Python 3.12 using different TTS libraries.
This avoids the Coqui TTS Python version compatibility issues.
"""

import argparse
from pathlib import Path
import asyncio
import edge_tts
from gtts import gTTS
import pyttsx3
from pydub import AudioSegment
import os

class TwiTTSAlternative:
    """Alternative Twi Text-to-Speech using Python 3.12 compatible libraries"""
    
    def __init__(self, engine='edge'):
        """
        Initialize TTS engine
        
        Args:
            engine: 'edge', 'gtts', or 'pyttsx3'
        """
        self.engine = engine
        
        if engine == 'pyttsx3':
            self.tts_engine = pyttsx3.init()
            # Try to set a voice (may not have Twi specifically)
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)
    
    async def synthesize_edge(self, text, output_path):
        """Synthesize using Edge TTS (Microsoft's cloud TTS)"""
        # Edge TTS has some African language support
        # Using English voice as fallback since Twi might not be available
        voice = "en-US-JennyNeural"  # You can experiment with other voices
        
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        return output_path
    
    def synthesize_gtts(self, text, output_path):
        """Synthesize using Google TTS"""
        # gTTS doesn't support Twi directly, using English as fallback
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_path)
        return output_path
    
    def synthesize_pyttsx3(self, text, output_path):
        """Synthesize using pyttsx3 (offline)"""
        self.tts_engine.save_to_file(text, output_path)
        self.tts_engine.runAndWait()
        return output_path
    
    def synthesize(self, text, output_path="output.wav"):
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            output_path: Output audio file path
        """
        print(f"Synthesizing with {self.engine}: '{text}'")
        
        try:
            if self.engine == 'edge':
                # Edge TTS requires async
                asyncio.run(self.synthesize_edge(text, output_path))
            elif self.engine == 'gtts':
                # gTTS saves as MP3, convert to WAV if needed
                temp_mp3 = output_path.replace('.wav', '.mp3')
                self.synthesize_gtts(text, temp_mp3)
                
                if output_path.endswith('.wav'):
                    # Convert MP3 to WAV
                    audio = AudioSegment.from_mp3(temp_mp3)
                    audio.export(output_path, format="wav")
                    os.remove(temp_mp3)
                else:
                    os.rename(temp_mp3, output_path)
            elif self.engine == 'pyttsx3':
                self.synthesize_pyttsx3(text, output_path)
            
            print(f"Audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error during synthesis: {e}")
            return None
    
    def batch_synthesize(self, texts, output_dir="outputs"):
        """Synthesize multiple texts"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"synthesis_{i:03d}.wav"
            result = self.synthesize(text, str(output_path))
            results.append(result)
        
        return results

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Alternative Twi Text-to-Speech for Python 3.12")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file")
    parser.add_argument("--engine", type=str, choices=['edge', 'gtts', 'pyttsx3'], 
                       default='edge', help="TTS engine to use")
    
    args = parser.parse_args()
    
    # Initialize TTS
    tts_model = TwiTTSAlternative(engine=args.engine)
    
    # Synthesize
    result = tts_model.synthesize(
        text=args.text,
        output_path=args.output
    )
    
    if result:
        print("Synthesis completed successfully!")
    else:
        print("Synthesis failed!")

def demo():
    """Demo function with sample Twi texts"""
    print("Running Alternative Twi TTS Demo for Python 3.12...")
    print("Note: These engines don't natively support Twi, so pronunciation may not be accurate.")
    
    # Sample Twi texts from your dataset
    sample_texts = [
        "Meda so ara wom.",
        "Ɛno na ɛboaa me.",
        "Dabi, ɛnte saa",
        "Yɛhyɛɛ no ma ogye toom sɛ ɔbɛyɛ.",
        "Woyɛ ɔbɔnefo."
    ]
    
    # Create demo outputs
    demo_dir = Path("demo_outputs_alt")
    demo_dir.mkdir(exist_ok=True)
    
    # Try different engines
    engines = ['edge', 'gtts', 'pyttsx3']
    
    for engine in engines:
        print(f"\n--- Testing {engine.upper()} engine ---")
        try:
            tts_model = TwiTTSAlternative(engine=engine)
            
            for i, text in enumerate(sample_texts[:2]):  # Test first 2 texts
                output_path = demo_dir / f"demo_{engine}_{i+1:02d}.wav"
                print(f"Synthesizing: {text}")
                result = tts_model.synthesize(text, str(output_path))
                if result:
                    print(f"✓ Success: {output_path}")
                else:
                    print(f"✗ Failed")
        except Exception as e:
            print(f"✗ {engine} engine failed: {e}")
    
    print(f"\nDemo complete! Check outputs in {demo_dir}/")
    print("Note: For proper Twi pronunciation, you'll need Python 3.11 with Coqui TTS")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Run demo if no arguments provided
        demo()
    else:
        # Run CLI
        main()