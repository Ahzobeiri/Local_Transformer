import torch
import torch.nn as nn
from .cbramod import CBraMod

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        # Change out_dim to 800 in the backbone.
        self.backbone = CBraMod(
            in_dim=200,
            out_dim=800,    # updated from 200 to 800
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8
        )

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        # Override the projection layer as before.
        self.backbone.proj_out = nn.Identity()

        # Updated flattened size: 18 channels * 30 time steps * 800 features = 432000
        flattened_size = 18 * 30 * 800

        # Updated classifier: added an extra linear layer block with more neurons.
        self.classifier_layer = nn.Sequential(
            nn.Linear(flattened_size, 10 * 800),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(10 * 800, 5 * 800),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(5 * 800, 800),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(800, 1) 
        )
    
    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, channels, seq_len, patch_size)
           Example: (batch_size, 18, 30, 200)
        """
        bz, ch_num, seq_len, patch_size = x.shape
        # Pass through the backbone.
        feats = self.backbone(x)
        # Flatten the features: note that the last dimension is now 800.
        out = feats.contiguous().view(bz, ch_num * seq_len * 800)
        # Pass through the classifier.
        out_classifier = self.classifier_layer(out)
        return out_classifier
