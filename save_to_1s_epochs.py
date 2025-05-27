import os
import lmdb
import pickle
import numpy as np
import scipy.signal
from tqdm import tqdm
from collections import defaultdict

LMDB_PATH   = '/path/to/LMDB_DATA'
OUTPUT_PATH = '/path/to/output/LEAD/ARREST'
feat_dir    = os.path.join(OUTPUT_PATH, 'Feature')
label_dir   = os.path.join(OUTPUT_PATH, 'Label')
os.makedirs(feat_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# 1) Load split keys
env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
with env.begin() as txn:
    splits = pickle.loads(txn.get(b'__keys__'))

all_keys = splits['train'] + splits['val'] + splits['test']

# 2) Group keys by patient_id prefix (before first '_')
by_patient = defaultdict(list)
for key in all_keys:
    pid = key.split('_')[0]  # e.g. 'P001'
    by_patient[pid].append(key)

patient_labels = []
# 3) Process each patient
with env.begin() as txn:
    for pid, keys in tqdm(by_patient.items(), desc="Patients"):
        all_epochs = []  # list of (n_i,128,19) arrays
        outcome = None

        for key in keys:
            raw = txn.get(key.encode())
            if raw is None:
                continue
            data_dict  = pickle.loads(raw)
            sample_30s = data_dict['sample']  # (19, 30, 200)
            outcome    = int(data_dict['outcome'])  # should be same for all chunks

            # collapse time: (19, 30*200)
            sig = sample_30s.reshape(19, -1)

            # resample 200→128 Hz
            sig128 = scipy.signal.resample_poly(sig, 128, 200, axis=1)

            # epoch into 1 s windows: each window = 128 samples
            total = sig128.shape[1]
            n_win = total // 128
            sig128 = sig128[:, :n_win*128]
            ep = sig128.reshape(19, n_win, 128).transpose(1, 2, 0)  # (n_win,128,19)

            all_epochs.append(ep)

        if not all_epochs:
            print(f"[WARN] no valid chunks for {pid}")
            continue

        # 4) concatenate all windows for this patient
        patient_data = np.vstack(all_epochs)  # shape (N_total,128,19)

        # save per-patient feature
        out_feat = os.path.join(feat_dir, f'feature_{pid}.npy')
        np.save(out_feat, patient_data.astype(np.float32))

        # record label
        patient_labels.append([pid, outcome])

# 5) Save patient-level label file
labels_arr = np.array(patient_labels, dtype=object)  # (n_patients,2)
np.save(os.path.join(label_dir, 'label.npy'), labels_arr)

print(f"Saved {len(patient_labels)} patients → {OUTPUT_PATH}")
