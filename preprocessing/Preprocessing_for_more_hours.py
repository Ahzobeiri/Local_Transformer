#!/usr/bin/env python

from helper_code import *
import numpy as np, os, sys
import mne
import joblib
import lmdb
import pickle
from tqdm import tqdm
import random
import scipy.signal

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_eeg(data_folder, patient_id, db: lmdb.Environment, file_key_list: list):
    """
    Load & concatenate the first `count` EEG recordings for a patient,
    then preprocess, epoch, and save each 30 s segment to LMDB.
    """
    # **** Extract labels for the patient ****
    # Load the patient metadata and extract the outcome and CPC labels.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    outcome = get_outcome(patient_metadata)  # Binary: 0 (Good) or 1 (Poor)
    cpc = get_cpc(patient_metadata)          # CPC score: integer (1-5)

    
    # Define all unique electrodes needed for the montage.
    eeg_channels = ['Fp1', 'F7', 'T3', 'T5', 'O1',
                    'Fp2', 'F8', 'T4', 'T6', 'O2',
                    'F3', 'C3', 'P3', 'F4', 'C4',
                    'P4', 'Fz', 'Cz', 'Pz']
    
    group = 'EEG'
    
    # 1. Find all recording files for the patient.
    recording_ids = find_recording_files(data_folder, patient_id)
    # Check if there are any recordings available.
    if len(recording_ids) == 0:
        return False

    count = 2
    
    # 2. pick the first subset of recordings we want
    selected = recording_ids[:count]


    all_data = []
    all_sf = []
    all_ut = []

    # 3.Load & Accumulate
    for recording_id in selected:
        recording_location = os.path.join(data_folder, patient_id, f'{recording_id}_{group}')
        
        # Check if the header file exists.
        if not os.path.exists(recording_location + '.hea'):
            return False

        # Load raw data
        data, channels, sampling_frequency = load_recording_data(recording_location)
        utility_frequency = get_utility_frequency(recording_location + '.hea')
        
        # Ensure all required EEG channels are available.
        if not all(channel in channels for channel in eeg_channels):
            print(f"Missing channels in {rec}, skipping")
            return False
            
        # Channel Processing: expand to ensure channels in correct order.
        data = expand_channels(data, channels, eeg_channels)
        channels = eeg_channels

        all_data.append(data)
        all_sf.append(sampling_frequency)
        all_ut.append(ut)

    if not all_data:
        return False

    '''
     # 4. sanity‐check sampling / utility freqs
     if len(set(all_sf)) != 1 or len(set(all_ut)) != 1:
         raise RuntimeError("Different sampling or utility freqs across recordings")
    '''

    # 5. concatenate in time
    data_cat = np.concatenate(all_data, axis=1)
    sf = all_sf[0]
    ut = all_ut[0]

    # 6. Preprocessing: filtering, notch and bandpass, then resampling.
    data_filt, new_sf = preprocess_data(data_cat, sf, ut)
    

    ######################################################
    # 7.Epoch Segmentation
    ######################################################
    # Transpose to (samples, channels) format.
    signal_data = data_filt.T  # Now shape (num_samples, num_channels=19)      
    total_samples = signal_data.shape[0]
    
    # Check minimum length (e.g., at least 2 minutes)
    if total_samples < 120 * 200:
        print("Not enough data after concatenation")
        return False
        
    a = total_samples % (30 * 200)
    # Trim data to remove unstable segments at start and end.
    trimmed_data = signal_data[60 * 200 : -(a + 60 * 200), :]
    
    # Reshape into epochs: each epoch is 30 seconds long, sampled at 128 Hz.
    num_epochs = trimmed_data.shape[0] // (30 * 200)
    segmented = trimmed_data.reshape(num_epochs, 30 * 200, 19)
    # Transpose to get final shape: (num_epochs, channels, time_steps, samples_per_second)
    segmented = segmented.reshape(num_epochs, 30, 200, 19).transpose(0, 3, 1, 2)  # (epochs, 19, 30, 200)
    print("Segmented data shape:", segmented.shape)
    
    '''
    # Normalize the segmented data.
    min_val = np.min(segmented)
    max_val = np.max(segmented)
    if min_val != max_val:
        segmented = 2 * (segmented - min_val) / (max_val - min_val) - 1
    '''
    
    ######################################################
    # Save each epoch along with labels to LMDB.
    ######################################################
    # Use the file name from the recording location for creating a unique key.
    file_name = f"{patient_id}_first_{count}"
    for i, sample in enumerate(segmented):
        sample_key = f'{file_name}_epoch{i}'
        print("Saving sample:", sample_key)
        file_key_list.append(sample_key)
        
        # Create a dictionary with the EEG sample and its labels.
        data_dict = {
            'sample': sample.astype(np.float32),
            'outcome': outcome,  # Binary: 0 (Good) or 1 (Poor)
            'cpc': cpc           # CPC score: integer (1-5)
        }
        with db.begin(write=True) as txn:
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            
    return True         

    except Exception as e:
        print(f"Error processing {patient_id}: {str(e)}")
        return False

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.3, 75.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)
    
   # Apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')
    
    # Resample the data.
    resampling_frequency = 200  # Desired resampling frequency.
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    data = scipy.signal.resample_poly(data, up, down, axis=1)
    
    return data, resampling_frequency

if __name__ == '__main__':
    setup_seed(42)
    data_folder = 'projects/scratch/fhajati/physionet.org/files/i-care/2.1/training'
    output_db = 'projects/scratch/fhajati/physionet.org/files/i-care/2.1/LMDB_DATA'
    
    # Initialize LMDB (adjust map_size as needed)
    env = lmdb.open(output_db, map_size=1099511627776)  # 1 TB
    # Create a dictionary to hold the sample keys for each split.
    dataset = {
        'train': list(),
        'val': list(),
        'test': list(),
    }
    
    # Get all patient IDs.
    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)
    print(f"Total patients: {num_patients}")
    
    # Split patients into train (first 400), val (next 100), and test (remaining 107).
    patients_dict = {
        'train': patient_ids[:400],
        'val': patient_ids[400:500],
        'test': patient_ids[500:]
    }
    
    # Process each patient and assign the returned sample keys into the appropriate split.
    for mode in ['train', 'val', 'test']:
        print(f"Processing {mode} patients...")
        for patient_id in tqdm(patients_dict[mode]):
            success = get_eeg(data_folder, patient_id, env, dataset[mode])
            if not success:
                print(f"Skipped patient {patient_id} in {mode} split")
    
    # Save final key dictionary to LMDB.
    with env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(dataset))
    env.close()
