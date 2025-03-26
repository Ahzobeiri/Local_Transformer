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
torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
```

This is used for internal tracking within PyTorch, specifically for logging the usage of PyTorch API calls. If this line is inside `TransformerEncoder`, it will generate:
`"torch.nn.modules.TransformerEncoder"` and If it's inside `TransformerEncoderLayer`, it will generate: `"torch.nn.modules.TransformerEncoderLayer"`.


```python
self.layers = _get_clones(encoder_layer, num_layers)

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
```

It calls `_get_clones` to create a list (`ModuleList`) of identical encoder layers. This function takes in a PyTorch module and an integer `N`, and returns a container (specifically, an `nn.ModuleList`) filled with `N` independent copies of the provided module. The function uses `copy.deepcopy(module)` inside a list comprehension. The deepcopy operation ensures that all parameters and internal states of the module are copied recursively. This means that each clone is a completely independent instance; changes to one clone won’t affect the others. The comment `# FIXME: copy.deepcopy() is not defined on nn.module` suggests that there might be concerns or limitations with using `copy.deepcopy()` directly on PyTorch modules in some contexts. Despite the comment, the current implementation uses it, likely because it works for the intended use case. The comment might be a reminder to revisit this approach or to handle specific edge cases where `deepcopy` might fail.

### Forward Method (`def forward`):

**Input:**

*`src`*: the input tensor

*`mask`* and *`src_key_padding_mask`*: optional masks for attention.

*`is_causal`*: an optional flag indicating if the mask is causal.

**Processing:**

```python
output = src
for mod in self.layers:
    output = mod(output, src_mask=mask)
if self.norm is not None:
    output = self.norm(output)
return output
```

The input `src` is passed sequentially through each encoder layer (using a simple loop over self.layers).

After all layers, if a normalization layer (norm) is provided, it is applied to the output.

**Output:**

The final transformed tensor.


## TransformerEncoderLayer Class
This class implements one layer of the Transformer encoder with two main sub-blocks:

**Self-Attention Block** – to let the model focus on different parts of the input.

**Feed-Forward Block** – a two-layer MLP to further transform the data.

### Constructor (`__init__`):

**Parameters:**

*`d_model`*: dimension of the model (feature size).

*`nhead`*: number of attention heads.

*`dim_feedforward`*: size of the hidden layer in the feed-forward network.

*`dropout`*: dropout rate used to prevent overfitting.

*`activation`*: the activation function; can be provided as a string (e.g., *`"relu"`* or *`"gelu"`*) or a callable.

*`layer_norm_eps`*: epsilon for numerical stability in layer normalization.

*`batch_first`*: flag indicating whether batch dimension comes first.

*`norm_first`*: determines if layer normalization is applied before (pre-norm) the sub-blocks.

*`bias`*, *`device`*, and *`dtype`*: standard parameters for linear layers and multihead attention.


**MultiheadAttention Setup:**

The layer defines two separate attention modules:

*`self.self_attn_s`*: operates on one half of the input features.

*`self.self_attn_t`*: operates on the other half.

The division is made by splitting the last dimension (feature dimension) into two halves (i.e. *`d_model`* // 2). The number of heads is also halved accordingly (*`nhead`* // 2).

**Feed-Forward Network Setup:**

Two linear layers (*`linear1`* and *`linear2`*) with a dropout in between, implementing a standard MLP block.

**Normalization and Dropout Layers:**

Two layer normalization layers (*`norm1`* and *`norm2`*) are defined.

Additional dropout layers (*`dropout1*` and *`dropout2*`) are used after each sub-block.

**Activation Function Handling:**

If the activation is provided as a string, *`_get_activation_fn`* converts it to the corresponding function.

There is logic to mark whether the activation is either ReLU or GELU.
