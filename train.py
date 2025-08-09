#!/usr/bin/env python3
"""
Training script for Coqui TTS Tacotron2 model on Twi language dataset.
"""

import os
import sys
from trainer import Trainer, TrainerArgs
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from config import config


def main():
    """Main training function"""
    print("Starting Coqui TTS training for Twi language...")
    # print(f"Output directory: {config.output_path}")

    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)

    # Initialize audio processor
    print("Initializing audio processor...")
    ap = AudioProcessor.init_from_config(config)

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Load dataset
    print("Loading dataset samples...")
    train_samples, eval_samples = load_tts_samples(
        config.datasets[0],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f"Training samples: {len(train_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")

    # Initialize model
    print("Initializing Tacotron2 model...")
    model = Tacotron2(config, ap, tokenizer, speaker_manager=None)

    # Trainer arguments
    trainer_args = TrainerArgs(
        restore_path=None,  # Set path to continue training from checkpoint
        skip_train_epoch=False,
        start_with_eval=False,
        grad_accum_steps=1,
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        trainer_args,
        config,
        output_path=config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Start training
    print("Starting training...")
    print("=" * 50)
    trainer.fit()

    print("Training completed!")
    print(f"Model saved in: {config.output_path}")


if __name__ == "__main__":
    main()
