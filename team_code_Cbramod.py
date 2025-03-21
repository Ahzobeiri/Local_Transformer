#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from block import FinalModel


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.


# Function to segment a signal into non-overlapping chunks of length 8192.
def segment_signal(signal, seg_length=8192):
    """
    Divide a 2-channel signal into non-overlapping segments of seg_length.

    Parameters:
      signal (np.array): EEG signal of shape (2, L).
      seg_length (int): Desired segment length.

    Returns:
      np.array: Array of segments with shape (num_segments, 2, seg_length)
    """
    num_samples = signal.shape[1]
    segments = []
    # Step through the signal in increments of seg_length.
    for start in range(0, num_samples - seg_length + 1, seg_length):
        seg = signal[:, start:start + seg_length]
        segments.append(seg)
    return np.array(segments)  # shape: (num_segments, 2, seg_length)



# New training function that uses a neural network and segments each EEG signal.
def train_challenge_model(data_folder, model_folder, verbose, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Find patient IDs.
    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)
    if num_patients == 0:
        raise FileNotFoundError('No data provided.')

    signals_list = []
    outcomes_list = []

    # For each patient, extract EEG signal, segment it, and assign the same outcome to all segments.
    for i, patient_id in enumerate(patient_ids):
        if verbose >= 2:
            print(f"Processing patient {i + 1}/{num_patients}...")

        # Get the EEG signal (assumed to have shape (2, L)).
        signal = get_eeg(data_folder, patient_id)
        if signal is None:
            continue

        # Segment the signal into chunks of length 8192.
        segments = segment_signal(signal, seg_length=8192)  # shape: (num_segments, 2, 8192)
        if segments.size == 0:
            continue

        # Expand dimensions to get input shape (batch, channels=2, 1, 8192)
        segments = np.expand_dims(segments, axis=2)  # now shape: (num_segments, 2, 1, 8192)
        signals_list.append(segments)

        # Extract the outcome for the patient (assumed scalar).
        patient_metadata = load_challenge_data(data_folder, patient_id)
        outcome = get_outcome(patient_metadata)
        # Replicate the outcome for each segment.
        outcomes_list.append(np.full((segments.shape[0], 1), outcome))

    if len(signals_list) == 0:
        raise ValueError("No valid EEG signals were found.")

    # Concatenate all segments and outcomes.
    X = np.concatenate(signals_list, axis=0)  # shape: (total_segments, 2, 1, 8192)
    y = np.concatenate(outcomes_list, axis=0)  # shape: (total_segments, 1)

    # Convert to PyTorch tensors.
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model, loss function, and optimizer.
    model = FinalModel(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Adjust if your outcome is categorical.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(dataset)
        if verbose >= 1:
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    # Create model folder if it doesn't exist and save the model.
    os.makedirs(model_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_folder, "eeg_model.pt"))
    if verbose >= 1:
        print("Training complete and model saved.")



def get_eeg(data_folder, patient_id):
    """
    Load and process the EEG signal for a given patient.

    Parameters:
      data_folder (str): The folder containing patient data.
      patient_id (str): The unique identifier for the patient.

    Returns:
      np.array or None: The preprocessed EEG signal in bipolar montage
                        or None if the required data is not available.
    """
    
    # Find recording files for the patient.
    recording_ids = find_recording_files(data_folder, patient_id)

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

    # Check if there are any recordings available.
    if len(recording_ids) > 0:
        # Use the most recent recording.
        recording_id = recording_ids[-1]
        recording_location = os.path.join(data_folder, patient_id, f'{recording_id}_{group}')

        # Check if the header file exists.
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            # Ensure all required EEG channels are available.
            if all(channel in channels for channel in eeg_channels):
                data, channels = expand_channels(data, channels, eeg_channels)
                data, channels = reduce_channels(data, channels, eeg_channels)
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                
                
                channel_indices = {channel: idx for idx, channel in enumerate(channels)}
                signal = np.array([
                    data[channel_indices[ch1], :] - data[channel_indices[ch2], :]
                    for ch1, ch2 in bipolar_montage
                ])

                
                ######################################################
                # NEW: Epoch Segmentation
                #####################################################
                # Transpose to (samples, channels) format
                signal = signal.T  # Now shape (num_samples, num_channels=18)

                total_samples = signal.shape[0]
                a = total_samples % (30 * 128)

                # Check minmum lenght (2 minutes)
                if total_samples < 120 * 128:
                    return


                trimmed_data = signal[60 * 128 : -(a + 60 * 128), :]
                # print(trimmed_data.shape)

        
                # Reshape into epochs
                num_epochs = trimmed_data.shape[0] // (30 * 128)
                segmented = trimmed_data.reshape(num_epochs, 30, 128, 18)
                segmented = segmented.transpose(0, 3, 1, 2) # Final shape: (num_epochs, channels, time_steps, samples_per_second)
                print(segmented.shape)

                
                # SCALE HERE (after segmentation)
                min_val = np.min(segmented)
                max_val = np.max(segmented)
                if min_val != max_val:
                segmented = 2 * (segmented - min_val) / (max_val - min_val) - 1

                file_name = recording_location.split('/')[-1]



                for i, sample in enumerate(segmented):
                    # print(i, sample.shape)
                    sample_key = f'{file_name}_{i}'
                    print(sample_key)
                    file_key_list.append(sample_key)
                    txn = db.begin(write=True)
                    txn.put(key=sample_key.encode(), value=pickle.dumps(sample))
                    txn.commit()
                
                
                
                return segmented
                
            else:
                return None
        else:
            return None
    else:
        return None

    return signal   ### (These Returns Should be checked!!!!)




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
    
'''
    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data
'''
    return data, resampling_frequency


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Extract features.
    features = get_features(data_folder, patient_id)
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)





# Extract features.
def get_features(data_folder, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.
    eeg_channels = ['F3', 'P3', 'F4', 'P4']
    group = 'EEG'

    if num_recordings > 0:
        recording_id = recording_ids[-1]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            if all(channel in channels for channel in eeg_channels):
                data, channels = reduce_channels(data, channels, eeg_channels)
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                data = np.array(
                    [data[0, :] - data[1, :], data[2, :] - data[3, :]])  # Convert to bipolar montage: F3-P3 and F4-P4
                eeg_features = get_eeg_features(data, sampling_frequency).flatten()
            else:
                eeg_features = float('nan') * np.ones(8)  # 2 bipolar channels * 4 features / channel
        else:
            eeg_features = float('nan') * np.ones(8)  # 2 bipolar channels * 4 features / channel
    else:
        eeg_features = float('nan') * np.ones(8)  # 2 bipolar channels * 4 features / channel

    # Extract ECG features.
    ecg_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
    group = 'ECG'

    if num_recordings > 0:
        recording_id = recording_ids[0]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            data, channels = reduce_channels(data, channels, ecg_channels)
            data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
            features = get_ecg_features(data)
            ecg_features = expand_channels(features, channels, ecg_channels).flatten()
        else:
            ecg_features = float('nan') * np.ones(10)  # 5 channels * 2 features / channel
    else:
        ecg_features = float('nan') * np.ones(10)  # 5 channels * 2 features / channel

    # Extract features.
    return np.hstack((patient_features, eeg_features, ecg_features))


# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male = 0
        other = 0
    elif sex == 'Male':
        female = 0
        male = 1
        other = 0
    else:
        female = 0
        male = 0
        other = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

    return features


# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=0.5, fmax=8.0,
                                                          verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=4.0, fmax=8.0,
                                                          verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=8.0, fmax=12.0,
                                                          verbose=False)
        beta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0,
                                                         verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean = np.nanmean(beta_psd, axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)

    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T

    return features


# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std = float('nan') * np.ones(num_channels)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features
