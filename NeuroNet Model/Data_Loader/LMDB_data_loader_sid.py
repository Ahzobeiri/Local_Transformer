import os
import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class LMDBChannelEpochDataset(Dataset):
    def __init__(self, lmdb_path: str, mode: str,
                 fs: int = 200, n_channels: int = 18):
        """
        lmdb_path: directory of your LMDB env
        mode: 'train' | 'val' | 'test'
        fs: sampling frequency (200)
        n_channels: number of EEG channels (18)
        """
        super().__init__()
        # Open the LMDB environment
        self.db = lmdb.open(
            lmdb_path,
            readonly=True, lock=False, readahead=True, meminit=False
        )

        # Load the split key-list
        with self.db.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get(b'__keys__'))
            self.keys = all_keys[mode]

        self.fs = fs
        self.n_ch = n_channels

        # KEY CHANGE: Call the _load_all method and unpack all four returned arrays
        self.total_x, self.total_y, self.total_sids, self.total_hospitals = self._load_all()

    def _load_all(self):
        """
        Loads all required data (samples, labels, subject IDs, hospital IDs)
        from the LMDB database into memory.
        """
        x_list = []
        y_list = []
        sid_list = []
        hospital_list = [] # KEY CHANGE: Add a list for hospital IDs

        for key in tqdm(self.keys, desc=f"Loading {len(self.keys)} '{self.keys.name}' samples"):
            with self.db.begin(write=False) as txn:
                data_dict = pickle.loads(txn.get(key.encode()))

            # IMPORTANT: This keeps the multi-channel sample intact, which is what our model needs.
            # It does NOT flatten the channels like your original version.
            x_list.append(data_dict['sample'])      # shape (18, 30, 200)
            y_list.append(data_dict['cpc'] - 1)     # map 1–5 → 0–4
            sid_list.append(data_dict['sid'])       # Load the subject ID
            
            # KEY CHANGE: Load the hospital ID.
            # This assumes 'hospital' is a key in your data_dict. See note below.
            hospital_list.append(data_dict['hospital'])

        # Stack lists into final numpy arrays
        total_x = np.array(x_list, dtype=np.float32)
        total_y = np.array(y_list, dtype=np.int64)
        total_sids = np.array(sid_list)
        total_hospitals = np.array(hospital_list) # KEY CHANGE: Convert hospital list to array

        return total_x, total_y, total_sids, total_hospitals

    def __len__(self):
        return len(self.total_y)

    def __getitem__(self, idx):
        # This method is here for completeness, but our training script
        # will access the .total_x, .total_y, etc. attributes directly.
        x = torch.tensor(self.total_x[idx], dtype=torch.float32)
        y = torch.tensor(self.total_y[idx], dtype=torch.int64)
        sid = self.total_sids[idx] # Note: sid and hospital are not typically returned as tensors
        hospital = self.total_hospitals[idx]
        
        return x, y, sid, hospital
