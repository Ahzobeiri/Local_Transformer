#!/usr/bin/env python

from helper_code import *
import numpy as np, os, sys
import mne
import joblib
import lmdb
import pickle
from tqdm import tqdm
import os
import random
import numpy as np



def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_eeg(data_folder, patient_id, db: lmdb.open, file_key_list: list):
    """
    Process and save segmented EEG epochs to LMDB for a given patient.

    Parameters:
      data_folder (str): The folder containing patient data.
      patient_id (str): The unique identifier for the patient.
    """

    # Specify the EEG channels of interest.
    # eeg_channels = ['F3', 'P3', 'F4', 'P4']
    bipolar_montage = [
    ('Fp1', 'F7'),    # Fp1-F7
    ('F7', 'T3'),     # F7-T3 (or F7-T7)
    ('T3', 'T5'),     # T3-T5 (or T7-P7)
    ('T5', 'O1'),     # T5-O1 (or P7-O1)
    ('Fp2', 'F8'),    # Fp2-F8
    ('F8', 'T4'),     # F8-T4 (or F8-T8)
    ('T4', 'T6'),     # T4-T6 (or T8-P8)
    ('T6', 'O2'),     # T6-O2 (or P8-O2)
    ('Fp1', 'F3'),    # Fp1-F3
    ('F3', 'C3'),     # F3-C3
    ('C3', 'P3'),     # C3-P3
    ('P3', 'O1'),     # P3-O1
    ('Fp2', 'F4'),    # Fp2-F4
    ('F4', 'C4'),     # F4-C4
    ('C4', 'P4'),     # C4-P4
    ('P4', 'O2'),     # P4-O2
    ('Fz', 'Cz'),     # Fz-Cz
    ('Cz', 'Pz')      # Cz-Pz
    ]
    
    
    # Define all unique electrodes needed for the montage
    eeg_channels = list({ch for pair in bipolar_montage for ch in pair})  # Auto-extract unique electrodes
    
    group = 'EEG'
    
    # Find recording files for the patient.
    recording_ids = find_recording_files(data_folder, patient_id)

    # Check if there are any recordings available.
    if len(recording_ids) == 0:
        return False
    
        
    # Use the most recent recording.
    recording_id = recording_ids[-1]
    recording_location = os.path.join(data_folder, patient_id, f'{recording_id}_{group}')

    # Check if the header file exists.
    if not os.path.exists(recording_location + '.hea'):
        return False

    try:
        # Load raw data
        data, channels, sampling_frequency = load_recording_data(recording_location)
        utility_frequency = get_utility_frequency(recording_location + '.hea')
        
        # Ensure all required EEG channels are available.
        if not all(channel in channels for channel in eeg_channels):
            return False
            
        # Channel Processing
        data = expand_channels(data, channels, eeg_channels)
        data, channels = reduce_channels(data, channels, eeg_channels)
        
        # Preprocessing
        data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
        
        # Bipolar Referencing
        channel_indices = {channel: idx for idx, channel in enumerate(channels)}
        signal = np.array([
                    data[channel_indices[ch1], :] - data[channel_indices[ch2], :]
                    for ch1, ch2 in bipolar_montage
        ])
        
        
        ######################################################
                    # Epoch Segmentation
        #####################################################
        # Transpose to (samples, channels) format
        signal = signal.T  # Now shape (num_samples, num_channels=18)
        
        total_samples = signal.shape[0]
        a = total_samples % (30 * 128)
        
        # Check minmum lenght (2 minutes)
        if total_samples < 120 * 128:
            return False
            
        trimmed_data = signal[60 * 128 : -(a + 60 * 128), :]
        # print(trimmed_data.shape)
        
        # Reshape into epochs
        num_epochs = trimmed_data.shape[0] // (30 * 128)
        segmented = trimmed_data.reshape(num_epochs, 30, 128, 18)
        segmented = segmented.transpose(0, 3, 1, 2) # Final shape: (num_epochs, channels, time_steps, samples_per_second)
        print(segmented.shape)

                
        # Normalize
        min_val = np.min(segmented)
        max_val = np.max(segmented)
        if min_val != max_val:
            segmented = 2 * (segmented - min_val) / (max_val - min_val) - 1
            # Save to LMDB
            file_name = recording_location.split('/')[-1]
            for i, sample in enumerate(segmented):
                # print(i, sample.shape)
                sample_key = f'{file_name}_epoch{i}'
                print(sample_key)
                file_key_list.append(sample_key)
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(sample.astype(np.float32)))
                txn.commit()
                
            return True         

    except Exception as e:
        print(f"Error processing {patient_id}: {str(e)}")
        return False




# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data
    resampling_frequency = 128  # Change based on the desired resampling_frequency
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    data = scipy.signal.resample_poly(data, up, down, axis=1)
    
    return data, resampling_frequency


if __name__ == '__main__':
    setup_seed(42)
    data_folder = 'path/to/data'
    output_db = 'path/to/database.lmdb'
    
    # Initialize LMDB
    env = lmdb.open(output_db, map_size=1099511627776)  # 1TB
    file_keys = []

    # Process all patients
    patient_ids = find_data_folders(data_folder)
    
    for patient_id in tqdm(patient_ids):
        success = get_eeg(data_folder, patient_id, env, file_keys)
        if not success:
            print(f"Skipped patient {patient_id}")

    # Save final key list
    with env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(file_keys))
    env.close()
