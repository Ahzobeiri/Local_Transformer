import os
import lmdb
import pickle
import numpy as np
import mne
from tqdm import tqdm

# --- User parameters ---
LMDB_PATH    = '/path/to/first_hours_lmdb'   # your LMDB directory
OUTPUT_PATH  = '/path/to/output/LEAD/ARREST'  # where to save Feature/Label
CHANNELS_19  = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T3','C3','Cz','C4','T4',
    'T5','P3','Pz','P4','T6','O1','O2'
]
SFREQ_TARGET = 128
FILTER_LO    = 0.5
FILTER_HI    = 45.0
EPOCH_DUR    = 1.0  # seconds
# ------------------------

def preprocess_signals(data_dict):
    """
    Given a dict with keys 'signals', 'ch_names', 'sfreq', build an MNE Raw,
    pick desired channels, resample, filter, epoch, and return epochs array
    of shape (n_epochs, n_channels, n_times).
    """
    raw_data   = data_dict['signals']        # (n_ch_total, n_samples)
    ch_names   = data_dict['ch_names']
    sfreq_orig = data_dict['sfreq']
    
    info = mne.create_info(ch_names, sfreq_orig, ch_types='eeg')
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    raw.set_montage('standard_1020', on_missing='ignore')
    
    # pick only our 19
    pick = [ch for ch in raw.ch_names if ch in CHANNELS_19]
    raw.pick_channels(pick)
    
    # resample + band‐pass
    raw.resample(SFREQ_TARGET, npad='auto')
    raw.filter(FILTER_LO, FILTER_HI, fir_design='firwin')
    
    # fixed‐length, non‐overlapping epochs of 1 s
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_DUR, overlap=0.0, preload=True
    )
    data = epochs.get_data()  # shape (n_epochs, 19, n_times=128)
    
    # per‐epoch channel‐wise z‐score
    mean = data.mean(axis=2, keepdims=True)
    std  = data.std (axis=2, keepdims=True)
    return (data - mean) / (std + 1e-6)

def main():
    # prepare output dirs
    feat_dir  = os.path.join(OUTPUT_PATH, 'Feature')
    label_dir = os.path.join(OUTPUT_PATH, 'Label')
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    labels = []
    
    # open LMDB
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for idx, (key, val) in enumerate(tqdm(cursor, desc="Patients")):
            patient_id = key.decode('utf-8')
            try:
                data_dict = pickle.loads(val)
            except Exception as e:
                print(f"[WARN] could not unpickle patient {patient_id}: {e}")
                continue
            
            # preprocess EEG only
            try:
                epochs = preprocess_signals(data_dict)
            except Exception as e:
                print(f"[ERROR] preprocessing failed for {patient_id}: {e}")
                continue
            
            # reorder to (n_epochs, 128, 19)
            epochs = epochs.transpose(0, 2, 1)
            
            # save features
            out_feat = os.path.join(feat_dir, f'feature_{patient_id}.npy')
            np.save(out_feat, epochs)
            print(f"[{patient_id}] → {os.path.basename(out_feat)}, shape={epochs.shape}")
            
            # extract label (0=good, 1=poor) from your metadata
            # here we assume data_dict['outcome'] is already 0/1
            label = int(data_dict.get('outcome', -1))
            if label not in (0,1):
                print(f"[WARN] invalid outcome for {patient_id}, skipping label")
                continue
            labels.append([label, int(patient_id)])
    
    # save unified label matrix
    labels_np = np.array(labels, dtype=int)
    np.save(os.path.join(label_dir, 'label.npy'), labels_np)
    print(f"Saved label matrix {labels_np.shape} to {label_dir}/label.npy")

if __name__ == '__main__':
    main()
