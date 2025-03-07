# Local_Transformer
## sement_signal.py

**Input:**

*signal (np.array):* EEG signal of shape `(num_channels, num_points)`.

*seg_length (int):* Desired segment length. (By deafualt `8192`)

**Output:**

*np.array:* Array of segments with shape `(num_segments, 2, seg_length)`



```python 
train_challenge_model(data_folder, model_folder, verbose)
```
