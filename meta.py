#!/usr/bin/env python3
"""
Metadata analysis and dataset statistics for Twi TTS dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import librosa
import numpy as np
from collections import Counter


def analyze_dataset():
    """Analyze the Common Voice Twi dataset"""
    print("Analyzing Common Voice Twi Dataset...")

    # Read the validated dataset
    df = pd.read_csv('validated.tsv', sep='\t', header=0)

    print(f"\n=== Dataset Overview ===")
    print(f"Total validated samples: {len(df)}")
    print(f"Unique speakers: {df['client_id'].nunique()}")

    # Vote distribution
    print(f"\n=== Vote Distribution ===")
    print(f"Average upvotes: {df['up_votes'].mean():.2f}")
    print(f"Average downvotes: {df['down_votes'].mean():.2f}")

    vote_counts = df['up_votes'].value_counts().sort_index()
    print("Upvote distribution:")
    for votes, count in vote_counts.items():
        print(f"  {votes} upvotes: {count} samples")

    # Text length analysis
    df['text_length'] = df['sentence'].str.len()
    print(f"\n=== Text Length Statistics ===")
    print(f"Average text length: {df['text_length'].mean():.1f} characters")
    print(f"Min text length: {df['text_length'].min()}")
    print(f"Max text length: {df['text_length'].max()}")
    print(f"Median text length: {df['text_length'].median():.1f}")

    # Demographics (if available)
    print(f"\n=== Demographics ===")
    if 'age' in df.columns:
        age_counts = df['age'].value_counts()
        print("Age distribution:")
        for age, count in age_counts.items():
            if pd.notna(age):
                print(f"  {age}: {count} samples")

    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        print("Gender distribution:")
        for gender, count in gender_counts.items():
            if pd.notna(gender):
                print(f"  {gender}: {count} samples")

    # Filter high-quality samples
    high_quality = df[df['up_votes'] >= 2]
    print(f"\n=== High-Quality Subset (≥2 upvotes) ===")
    print(
        f"High-quality samples: {len(high_quality)} ({len(high_quality)/len(df)*100:.1f}%)")

    # Most common words
    print(f"\n=== Most Common Words ===")
    all_text = ' '.join(df['sentence'].fillna(''))
    words = all_text.split()
    word_counts = Counter(words)

    print("Top 10 most common words:")
    for word, count in word_counts.most_common(10):
        print(f"  '{word}': {count} occurrences")

    return df, high_quality


def analyze_audio_files(sample_size=100):
    """Analyze audio file characteristics"""
    print(f"\n=== Audio Analysis (sample of {sample_size} files) ===")

    clip_dir = Path('./clips')
    audio_files = list(clip_dir.glob('*.mp3'))[:sample_size]

    durations = []
    sample_rates = []

    for audio_file in audio_files:
        try:
            audio, sr = librosa.load(audio_file, sr=None)
            duration = len(audio) / sr
            durations.append(duration)
            sample_rates.append(sr)
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")

    if durations:
        print(f"Average duration: {np.mean(durations):.2f} seconds")
        print(f"Min duration: {np.min(durations):.2f} seconds")
        print(f"Max duration: {np.max(durations):.2f} seconds")
        print(f"Median duration: {np.median(durations):.2f} seconds")

        print(f"Sample rates: {set(sample_rates)}")

    return durations, sample_rates


def create_visualizations(df):
    """Create visualizations of the dataset"""
    print("\n=== Creating Visualizations ===")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Vote distribution
    axes[0, 0].hist(df['up_votes'], bins=range(
        0, df['up_votes'].max()+2), alpha=0.7)
    axes[0, 0].set_title('Distribution of Upvotes')
    axes[0, 0].set_xlabel('Number of Upvotes')
    axes[0, 0].set_ylabel('Frequency')

    # Text length distribution
    axes[0, 1].hist(df['text_length'], bins=50, alpha=0.7)
    axes[0, 1].set_title('Distribution of Text Lengths')
    axes[0, 1].set_xlabel('Text Length (characters)')
    axes[0, 1].set_ylabel('Frequency')

    # Quality vs text length
    quality_data = df[df['up_votes'] >= 2]['text_length']
    all_data = df['text_length']

    axes[1, 0].hist([all_data, quality_data], bins=30, alpha=0.7,
                    label=['All samples', 'High quality (≥2 upvotes)'])
    axes[1, 0].set_title('Text Length: All vs High Quality')
    axes[1, 0].set_xlabel('Text Length (characters)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # Speaker contribution
    speaker_counts = df['client_id'].value_counts()
    axes[1, 1].hist(speaker_counts, bins=30, alpha=0.7)
    axes[1, 1].set_title('Speaker Contribution Distribution')
    axes[1, 1].set_xlabel('Samples per Speaker')
    axes[1, 1].set_ylabel('Number of Speakers')

    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as 'dataset_analysis.png'")

    plt.show()


if __name__ == "__main__":
    # Run analysis
    df, high_quality = analyze_dataset()
    durations, sample_rates = analyze_audio_files()
    create_visualizations(df)

    print(f"\n=== Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"High-quality samples: {len(high_quality)}")
    print(f"Recommended for training: {len(high_quality)} samples")
    print(f"Analysis complete!")
