import numpy as np
from datasets.dataset_for_finetuning import CustomDataset  # Adjust the import as needed
from utils.util import to_tensor  # Adjust the import path if needed

def test_dataset(data_dir):
    # Instantiate the dataset in "train" mode.
    dataset = CustomDataset(data_dir, mode='train')
    print("Number of samples in train:", len(dataset))
    
    # Retrieve and print the first sample.
    sample, cpc = dataset[0]
    print("First sample shape:", np.array(sample).shape)
    print("First CPC label (after 0-indexing):", cpc)
    
    # Create a mini-batch using the collate function.
    batch_size = 4 if len(dataset) >= 4 else len(dataset)
    batch = [dataset[i] for i in range(batch_size)]
    x_data, cpc_labels = dataset.collate(batch)
    
    print("Collated batch x_data shape:", x_data.shape)
    print("Collated batch CPC labels:", cpc_labels)

if __name__ == '__main__':
    # Set this to the directory containing your LMDB files (data.mdb and lock.mdb)
    data_dir = "/projects/scratch/fhajati/physionet.org/files/i-care/2.1/LMDB_DATA"
    test_dataset(data_dir)
