import numpy as np
import librosa
from audio import preprocess_wav, wav_to_mel_spectrogram
from pathlib import Path
from params_data import *
from pypinyin import pinyin
from pypinyin import Style


def _process_utterance(wav: np.ndarray, speaker: str, text: str, out_dir: Path, base_name: str, skip_exsiting: bool):

    mel_fpath = out_dir.joinpath('mel', 'mel-%s.npy' % base_name)
    wav_fpath = out_dir.joinpath('audio', 'audio-%s.npy' % base_name)
    if skip_exsiting and mel_fpath.exists() and wav_fpath.exists():
        return None

    mel_spectrogram = wav_to_mel_spectrogram(wav)

    np.save(mel_fpath, mel_spectrogram)
    np.save(wav_fpath, wav)

    return mel_fpath.name, wav_fpath.name, "embedding-%s.npy" % base_name, speaker,\
           text, mel_spectrogram.shape[0], len(wav)

def _get_wav_text(fwav, words):
    wav = preprocess_wav(fwav)

    text = pinyin(words, style=Style.TONE3)
    text = [c[0] for c in text if c[0].strip()]
    text = ' '.join(text)
    return wav, text


def preprocess_speaker_general(speaker_dir: Path, out_dir: Path, skip_existing: bool, data_info: dict):
    metadata = []
    extensions = ["*.wav", "*.flac", "*.mp3"]

    for extension in extensions:
        wav_file_list = speaker_dir.glob(extension)
        for wav_file in wav_file_list:
            words = data_info.get(wav_file.name.split('.')[0])
            words = data_info.get(wav_file.name) if not words else words # 如果使用的名称带有文件类型
            if not words:
                print("No words")
                continue
            sub_basename = "%s_%02d" % (wav_file.name, 0)
            wav, text = _get_wav_text(wav_file, words)

            metadata.append(_process_utterance(wav, text, speaker_dir.name, out_dir, sub_basename, skip_existing))

    return [m for m in metadata if m is not None]