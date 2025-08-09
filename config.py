import os
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config

# Dataset configuration
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata_train.csv",
    meta_file_val="metadata_val.csv",
    path="./twi_dataset/",
    language="tw"
)

# Model configuration
config = Tacotron2Config(
    # Model
    model="tacotron2",

    # Training
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    save_step=1000,
    checkpoint=True,
    keep_all_best=True,
    keep_after=10000,

    # Text processing
    text_cleaner="phoneme_cleaners",
    use_phonemes=False,  # Set to True if you want phoneme-based training
    phoneme_language="en-us",  # Change if Twi phonemes are available
    phoneme_cache_path="./phoneme_cache",

    # Logging and evaluation
    print_step=25,
    print_eval=True,
    tb_model_param_stats=True,

    # Optimization
    mixed_precision=False,
    lr=1e-3,
    grad_clip=1.0,

    # Directories
    output_path="./training_output/",

    # Dataset
    datasets=[dataset_config],

    # Preprocessing
    min_text_len=1,
    max_text_len=500,
    min_audio_len=1 * 22050,  # 1 second
    max_audio_len=10 * 22050,  # 10 seconds
)

# Audio configuration
config.audio.sample_rate = 22050
config.audio.hop_length = 256
config.audio.win_length = 1024
config.audio.n_fft = 1024
config.audio.mel_fband = 80
config.audio.mel_fmin = 0
config.audio.mel_fmax = 8000
config.audio.do_trim_silence = True
config.audio.trim_db = 23
config.audio.norm_gain_db = None
