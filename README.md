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


# Criss_cross_transformer

## TransformerEncoder Class
This class defines a stack of encoder layers (stacks multiple `TransformerEncoderLayer` instances.). It is responsible for sequentially applying multiple encoder layers to an input.

### Constructor (`__init__`):

**Parameters:**

*`encoder_layer`*: a single encoder layer module instance.

*`num_layers`*: number of times the encoder layer should be cloned (stacked).

*`norm`*: an optional normalization layer (e.g., `LayerNorm`) applied after all layers.

*`enable_nested_tensor`* and *mask_check*: parameters that responsible for tensor handling and mask validation.

**Implementation Details:**

```python
self.layers = _get_clones(encoder_layer, num_layers)

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
```

It calls `_get_clones` to create a list (`ModuleList`) of identical encoder layers.


It logs API usage (a typical internal detail for tracking module instantiation in PyTorch).
