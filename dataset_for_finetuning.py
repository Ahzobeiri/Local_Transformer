import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        # Load the global key dictionary and select the keys for the desired split.
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        # Retrieve the data dictionary corresponding to the key.
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']        # EEG sample data
        # outcome = pair['outcome']    # Binary label (0 or 1)
        cpc = pair['cpc'] - 1          # Convert CPC Labels from [1,5] to [0,4]
        return data, cpc #, outcome

    def collate(self, batch):
        # Collate the batch into separate NumPy arrays.
        x_data = np.array([x[0] for x in batch])
        # outcome_labels = np.array([x[1] for x in batch])
        cpc_labels = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(cpc_labels).long() # to_tensor(outcome_labels).long() 


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        # Create a CustomeDataset for each splite
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
