import torch
import torch.nn as nn
from .cbramod import CBraMod

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        # Keep backbone unchanged (critical for compatibility)
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()

        # Maintain original flattened size calculation
        flattened_size = 18 * 30 * 200  # 108000 (preserve dimensions)

        # Simplified classifier with reduced layers
        self.classifier_layer = nn.Sequential(
            nn.Linear(flattened_size, 200),  # Direct dimension reduction
            nn.ReLU(),  # Simpler activation
            nn.Dropout(param.dropout),
            nn.Linear(200, 1)  # Single output layer
        )

    def forward(self, x):
        # Preserve original input handling
        bz, ch_num, seq_len, patch_size = x.shape
        
        # Maintain backbone processing flow
        feats = self.backbone(x)
        out = feats.contiguous().view(bz, -1)  # Keep original flattening
        
        return self.classifier_layer(out)
