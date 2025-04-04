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
    Process and save segmented EEG epochs to LMDB for a given patient.
    Also extract and store the corresponding outcome and CPC labels.
    
    Parameters:
      data_folder (str): The folder containing patient data.
      patient_id (str): The unique identifier for the patient.
      db (lmdb.Environment): Open LMDB environment for storing data.
      file_key_list (list): A list to collect keys for saved samples (for the current split).
    """
    
    # Specify the bipolar montage (pairs of channels)
    bipolar_montage = [
        ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
        ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
        ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
        ('Fz', 'Cz'), ('Cz', 'Pz')
    ]
    
    # Define all unique electrodes needed for the montage.
    eeg_channels = ['Fp1', 'F7', 'T3', 'T5', 'O1',
                    'Fp2', 'F8', 'T4', 'T6', 'O2',
                    'F3', 'C3', 'P3', 'F4', 'C4',
                    'P4', 'Fz', 'Cz', 'Pz']
    
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
        # **** Extract labels for the patient ****
        # Load the patient metadata and extract the outcome and CPC labels.
        patient_metadata = load_challenge_data(data_folder, patient_id)
        outcome = get_outcome(patient_metadata)  # Binary: 0 (Good) or 1 (Poor)
        cpc = get_cpc(patient_metadata)          # CPC score: integer (1-5)

        # Load raw data
        data, channels, sampling_frequency = load_recording_data(recording_location)
        utility_frequency = get_utility_frequency(recording_location + '.hea')
        
        # Ensure all required EEG channels are available.
        if not all(channel in channels for channel in eeg_channels):
            return False
            
        # Channel Processing: expand to ensure channels in correct order.
        data = expand_channels(data, channels, eeg_channels)
        channels = eeg_channels
        
        # Preprocessing: filtering, notch and bandpass, then resampling.
        data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
        
        # Bipolar Referencing: compute difference between paired channels.
        channel_indices = {channel: idx for idx, channel in enumerate(channels)}
        signal_data = np.array([
            data[channel_indices[ch1], :] - data[channel_indices[ch2], :]
            for ch1, ch2 in bipolar_montage
        ])
        
        ######################################################
        # Epoch Segmentation
        ######################################################
        # Transpose to (samples, channels) format.
        signal_data = signal_data.T  # Now shape (num_samples, num_channels=18)
        
        total_samples = signal_data.shape[0]
        a = total_samples % (30 * 128)
        
        # Check minimum length (e.g., at least 2 minutes)
        if total_samples < 120 * 128:
            return False
            
        # Trim data to remove unstable segments at start and end.
        trimmed_data = signal_data[60 * 128 : -(a + 60 * 128), :]
        
        # Reshape into epochs: each epoch is 30 seconds long, sampled at 128 Hz.
        num_epochs = trimmed_data.shape[0] // (30 * 128)
        segmented = trimmed_data.reshape(num_epochs, 30, 128, 18)
        # Transpose to get final shape: (num_epochs, channels, time_steps, samples_per_second)
        segmented = segmented.transpose(0, 3, 1, 2)
        print("Segmented data shape:", segmented.shape)

        # Normalize the segmented data.
        min_val = np.min(segmented)
        max_val = np.max(segmented)
        if min_val != max_val:
            segmented = 2 * (segmented - min_val) / (max_val - min_val) - 1

        ######################################################
        # Save each epoch along with labels to LMDB.
        ######################################################
        # Use the file name from the recording location for creating a unique key.
        file_name = recording_location.split('/')[-1]
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
            
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
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

    # Resample the data.
    resampling_frequency = 128  # Desired resampling frequency.
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
