import os
import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class LMDBChannelEpochDataset(Dataset):
    def __init__(self, lmdb_path: str, mode: str,
                 fs: int = 200, n_channels: int = 19):
        """
        lmdb_path: directory of your LMDB env
        mode: 'train' | 'val' | 'test'
        fs: sampling frequency (128)
        n_channels: number of EEG channels (19)
        """
        super().__init__()
        # open LMDB
        self.db = lmdb.open(
            lmdb_path,
            readonly=True, lock=False, readahead=True, meminit=False
        )
        # load the split key‐list
        with self.db.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get(b'__keys__'))
            self.keys = all_keys[mode]

        self.fs = fs
        self.n_ch = n_channels
        # build in‐memory arrays just like the parquet loader did
        self.total_x, self.total_y = self._load_all()

    def _load_all(self):
        tx_list = []
        ty_list = []
        for key in tqdm(self.keys, desc=f"Loading {len(self.keys)} samples"):
            with self.db.begin(write=False) as txn:
                # FIXED: key is already bytes
                data_dict = pickle.loads(txn.get(key))
            sample = data_dict['sample']      # Original shape: (19, 30, fs)
            cpc    = data_dict['cpc'] - 1      # Map 1–5 → 0–4
            
            # Reshape the (19, 30, fs) sample into (19, 30 * fs)
            n_channels, n_windows, fs = sample.shape
            reshaped_sample = sample.reshape(n_channels, n_windows * fs)
            
            tx_list.append(reshaped_sample)
            ty_list.append(cpc)
            
        # stack into arrays
        total_x = np.stack(tx_list, axis=0)
        total_y = np.array(ty_list, dtype=np.int64)
        return total_x, total_y

    def __len__(self):
        return len(self.total_y)

    def __getitem__(self, idx):
        x = torch.tensor(self.total_x[idx], dtype=torch.float32)
        y = torch.tensor(self.total_y[idx], dtype=torch.int64)
        return x, y
