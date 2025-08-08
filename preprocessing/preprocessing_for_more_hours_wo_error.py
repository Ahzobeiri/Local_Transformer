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
    try:
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
        count = 48 # As per your description of processing 48 hours
        
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
                print(f"Header file not found for {recording_location}, skipping recording.")
                continue

            try:
                # Load raw data
                data, channels, sampling_frequency = load_recording_data(recording_location)
                utility_frequency = get_utility_frequency(recording_location + '.hea')
            except ValueError as e:
                print(f"!!!!!! FAILED to load recording {recording_id} for patient {patient_id}. Error: {e}")
                print("!!!!!! This file may be corrupted. Skipping this specific recording.")
                continue # Skip to the next recording_id
            
            # Ensure all required EEG channels are available.
            if not all(channel in channels for channel in eeg_channels):
                print(f"Missing channels in {recording_id}, skipping")
                continue
                
            # Channel Processing: expand to ensure channels in correct order.
            data = expand_channels(data, channels, eeg_channels)
            channels = eeg_channels
            all_data.append(data)
            all_sf.append(sampling_frequency)
            all_ut.append(utility_frequency) # Corrected typo from 'ut'

        if not all_data:
            print(f"No valid recordings found for patient {patient_id} after checks.")
            return False
        
        # 4. sanity‐check sampling / utility freqs
        if len(set(all_sf)) > 1 or len(set(all_ut)) > 1:
             print(f"Warning: Different sampling or utility freqs for patient {patient_id}. Using first one.")

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
        if total_samples < 120 * new_sf: # Use new_sf
            print(f"Not enough data for patient {patient_id} after concatenation and preprocessing.")
            return False
            
        a = total_samples % (30 * new_sf)
        # Trim data to remove unstable segments at start and end.
        trimmed_data = signal_data[60 * new_sf : -(a + 60 * new_sf), :]
        
        # Reshape into epochs: each epoch is 30 seconds long.
        num_epochs = trimmed_data.shape[0] // (30 * new_sf)
        if num_epochs == 0:
            print(f"Not enough data to create even one epoch for patient {patient_id}.")
            return False
            
        segmented = trimmed_data.reshape(num_epochs, 30 * new_sf, 19)
        # Transpose to get final shape: (num_epochs, channels, time_steps)
        segmented = segmented.transpose(0, 2, 1)  # (epochs, 19, 6000)
        print("Segmented data shape:", segmented.shape)
        
        ######################################################
        # Save each epoch along with labels to LMDB.
        ######################################################
        file_name = f"{patient_id}_first_{count}"
        for i, sample in enumerate(segmented):
            sample_key = f'{file_name}_epoch{i}'
            # print("Saving sample:", sample_key) # Optional: can be noisy
            file_key_list.append(sample_key)
            
            data_dict = {
                'sample': sample.astype(np.float32),
                'outcome': outcome,
                'cpc': cpc
            }
            with db.begin(write=True) as txn:
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                
        return True
         
    except Exception as e:
        print(f"An unexpected error occurred while processing patient {patient_id}: {str(e)}")
        import traceback
        traceback.print_exc()
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
    
    dataset = {
        'train': list(),
        'val': list(),
        'test': list(),
    }
    
    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)
    print(f"Total patients: {num_patients}")
    
    patients_dict = {
        'train': patient_ids[:400],
        'val': patient_ids[400:500],
        'test': patient_ids[500:]
    }
    
    for mode in ['train', 'val', 'test']:
        print(f"Processing {mode} patients...")
        for patient_id in tqdm(patients_dict[mode]):
            try:
                success = get_eeg(data_folder, patient_id, env, dataset[mode])
                if not success:
                    print(f"Skipped patient {patient_id} in {mode} split (handled within function).")
            except Exception as e:
                print(f"!!!!!! CRITICAL ERROR processing patient {patient_id}. The function crashed unexpectedly. Error: {e}")
                print("!!!!!! Moving to the next patient.")

    # Save final key dictionary to LMDB.
    with env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(dataset))
    env.close()
    print("Processing complete.")
