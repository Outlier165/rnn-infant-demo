# src/data_utils.py
# Utilities for audio preprocessing: load, transform to log-mel, batch processing and simple NPZ dataset.

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

def load_and_transform(path: str,
                       sr: int = 16000,
                       max_len_secs: float = 1.0,
                       n_mels: int = 64,
                       n_fft: int = 1024,
                       hop_length: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load audio, pad/trim to max_len_secs, compute mel spectrogram (dB) and per-sample normalized spectrogram.
    Returns: (wav_padded, spec_db, spec_norm) where spec shape = (n_mels, time_frames).
    """
    wav, _ = librosa.load(path, sr=sr, mono=True)
    max_len = int(max_len_secs * sr)
    if len(wav) < max_len:
        wav = np.pad(wav, (0, max_len - len(wav)))
    else:
        wav = wav[:max_len]
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    return wav.astype(np.float32), S_db.astype(np.float32), S_norm.astype(np.float32)

def batch_process(file_list: List[str],
                  out_npz_path: str,
                  sr: int = 16000,
                  max_len_secs: float = 1.0,
                  n_mels: int = 64,
                  n_fft: int = 1024,
                  hop_length: int = 256,
                  verbose: bool = True,
                  max_files: int = None):
    """
    Process files in file_list, produce npz with X (N, n_mels, time_frames) and filenames list.
    """
    X = []
    names = []
    if max_files is not None:
        file_list = file_list[:max_files]
    for i, p in enumerate(file_list):
        try:
            _, _, spec_norm = load_and_transform(p, sr=sr, max_len_secs=max_len_secs,
                                                 n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
            X.append(spec_norm)
            names.append(os.path.basename(p))
        except Exception as e:
            if verbose:
                print(f"Skip {p}: {e}")
        if verbose and (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(file_list)}")
    if len(X) == 0:
        raise RuntimeError("No valid files processed.")
    X = np.stack(X)  # (N, n_mels, time_frames)
    os.makedirs(os.path.dirname(out_npz_path) or ".", exist_ok=True)
    np.savez_compressed(out_npz_path, X=X, files=names)
    if verbose:
        print(f"Saved {out_npz_path} with X.shape={X.shape}")
    return out_npz_path

def save_spec_png(spec_db: np.ndarray, out_png: str, sr: int = 16000, hop_length: int = 256, title: str = "Log-Mel Spectrogram"):
    """
    Save a spectrogram (in dB) as PNG.
    """
    plt.figure(figsize=(8,3))
    librosa.display.specshow(spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png

class NPZDataset(Dataset):
    """
    Simple Dataset to load precomputed spectrograms saved in npz by batch_process.
    Expects arrays: X (N, n_mels, time_frames) and files list.
    """
    def __init__(self, npz_path: str, transform=None):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data['X']
        # 'files' may be stored as object array
        self.files = data['files'] if 'files' in data else np.array([str(i) for i in range(len(self.X))])
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # shape (n_mels, time_frames)
        if self.transform:
            x = self.transform(x)
        # add channel dim for conv networks: (1, n_mels, time_frames)
        x = np.expand_dims(x, axis=0).astype(np.float32)
        return torch.from_numpy(x), idx
