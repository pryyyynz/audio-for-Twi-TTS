import pandas as pd
import os
import shutil
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm


def ensure_single_wav_extension(filename):
    """Ensure filename has only one .wav extension"""
    # Remove all .wav extensions
    base = filename.replace('.wav', '')
    # Add exactly one .wav extension
    return f"{base}.wav"


def prepare_coqui_dataset():
    """
    Prepare the Common Voice Twi dataset for Coqui TTS training.
    Converts audio files and creates metadata in the required format.
    """
    print("Preparing Coqui TTS dataset from Common Voice Twi data...")

    # Read the validated dataset
    df = pd.read_csv('validated.tsv', sep='\t', header=0)
    print(f"Found {len(df)} validated samples")

    # Filter for higher quality samples (rating >= 3)
    # Use samples with at least 2 upvotes
    df_filtered = df[df['up_votes'] >= 2]
    print(f"Using {len(df_filtered)} high-quality samples (up_votes >= 2)")

    # Create directory structure for Coqui TTS
    dataset_dir = Path('./twi_dataset')
    wavs_dir = dataset_dir / 'wavs'
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metadata
    metadata = []
    successful_conversions = 0

    print("Converting audio files and preparing metadata...")
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        try:
            audio_file = row['path']  # e.g., 'common_voice_tw_35292447.mp3'
            text = row['sentence'].strip()

            # Skip empty sentences
            if not text:
                continue

            # Source and destination paths
            src_path = f'./clips/{audio_file}'

            if os.path.exists(src_path):
                # Convert filename to match Coqui TTS convention
                # Ensure clean filename without double extensions
                base_name = f"tw_{successful_conversions:06d}"
                wav_filename = ensure_single_wav_extension(base_name)
                dst_path = wavs_dir / wav_filename

                # Load and convert audio to 22050Hz WAV
                try:
                    audio, sr = librosa.load(src_path, sr=22050)
                    # Normalize audio
                    audio = librosa.util.normalize(audio)
                    # Ensure the destination path has only one .wav extension
                    sf.write(str(dst_path), audio, 22050)

                    # Clean text (remove special characters that might cause issues)
                    clean_text = text.replace('"', '').replace(
                        '\n', ' ').replace('\r', ' ')
                    # Remove extra whitespace
                    clean_text = ' '.join(clean_text.split())

                    # Add to metadata (format: filename|text|text)
                    # Ensure filename has only one .wav extension
                    clean_wav_filename = ensure_single_wav_extension(wav_filename)
                    
                    metadata.append(
                        f"{clean_wav_filename}|{clean_text}|{clean_text}")
                    successful_conversions += 1

                except Exception as audio_error:
                    print(
                        f"Error processing audio {audio_file}: {audio_error}")
                    continue
            else:
                print(f"Audio file not found: {src_path}")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    # Write metadata file
    metadata_path = dataset_dir / 'metadata.csv'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(metadata))

    print(f"Dataset preparation complete!")
    print(f"Successfully converted {successful_conversions} audio files")
    print(f"Dataset location: {dataset_dir}")
    print(f"Audio files: {wavs_dir}")
    print(f"Metadata file: {metadata_path}")

    # Create train/val split
    split_dataset(metadata, dataset_dir)

    return dataset_dir


def split_dataset(metadata, dataset_dir):
    """Split dataset into train and validation sets (90/10 split)"""
    import random

    random.seed(42)  # For reproducible splits
    random.shuffle(metadata)

    split_idx = int(len(metadata) * 0.9)
    train_metadata = metadata[:split_idx]
    val_metadata = metadata[split_idx:]

    # Write train metadata
    with open(dataset_dir / 'metadata_train.csv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_metadata))

    # Write validation metadata
    with open(dataset_dir / 'metadata_val.csv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_metadata))

    print(f"Train samples: {len(train_metadata)}")
    print(f"Validation samples: {len(val_metadata)}")


if __name__ == "__main__":
    prepare_coqui_dataset()
