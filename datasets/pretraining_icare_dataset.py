import pickle
import lmdb
from torch.utils.data import Dataset
from utils.util import to_tensor

class IcarePretrainingDataset(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.db = lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            keys_dict = pickle.loads(txn.get('__keys__'.encode()))
            # Use all data for pretraining (train + val + test)
            self.keys = keys_dict['train'] + keys_dict['val'] + keys_dict['test']

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
            sample = pair['sample']  # Only return the EEG sample
        return to_tensor(sample)
