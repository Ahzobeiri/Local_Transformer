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
            '''
            # Compute class distribution if training set
            if mode == 'train':
            self.class_counts = self._compute_class_counts()
            print(f"Training set class counts: {self.class_counts}")
            '''
'''
    def _compute_class_counts(self):
        counts = [0] * 5  # 5 classes (0 to 4)
        with self.db.begin(write=False) as txn:
            for key in self.keys:
                pair = pickle.loads(txn.get(key.encode()))
                cpc = pair['cpc'] - 1  # Adjust to [0,4]
                counts[cpc] += 1
        return counts
    '''

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        # Retrieve the data dictionary corresponding to the key.
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']        # EEG sample data
        outcome = pair['outcome']    # Binary label (0 or 1)
        # cpc = pair['cpc'] - 1          # Convert CPC Labels from [1,5] to [0,4]
        return data, outcome

    def collate(self, batch):
        # Collate the batch into separate NumPy arrays.
        x_data = np.array([x[0] for x in batch])
        outcome_labels = np.array([x[1] for x in batch])
        # cpc_labels = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(outcome_labels).long() # to_tensor(cpc_labels).long() 

class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir
'''
        # Pass class counts to params if training
        self.train_set = CustomDataset(self.datasets_dir, mode='train')
        if hasattr(self.train_set, 'class_counts'):
            total_samples = sum(self.train_set.class_counts)
            num_classes = 5
            class_weights = [total_samples / (num_classes * count) if count > 0 else 1.0 
                            for count in self.train_set.class_counts]
            params.class_weights = torch.tensor(class_weights, dtype=torch.float)
'''
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
