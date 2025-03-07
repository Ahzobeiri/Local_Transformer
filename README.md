# Local_Transformer
## segment_signal

**Input:**

*signal (np.array):* EEG signal of shape `(num_channels, num_points)`.

*seg_length (int):* Desired segment length. (By deafualt `8192`)

**Output:**

*np.array:* Array of segments with shape `(num_segments, num_channels, seg_length)`


## train_challenge_model_nn

**Potential Checks**

```python
    signals_list = [] 
    outcomes_list = []
    cpcs = list()  # It should be added

    # For each patient, extract EEG signal, segment it, and assign the same outcome to all segments.
    for i, patient_id in enumerate(patient_ids):
        if verbose >= 2:
            print(f"Processing patient {i + 1}/{num_patients}...")

        # Get the EEG signal (assumed to have shape (2, L)).
        signal = get_eeg(data_folder, patient_id)
        if signal is None:
            continue
```
