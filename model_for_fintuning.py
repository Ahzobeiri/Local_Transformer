import torch
import torch.nn as nn

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()

        # For custom data: 18 channels, 30 time steps, backbone outputs dimension 200
        flattened_size = 18 * 30 * 200 # = 108000

        # Classifier for Outcome (binary: 2 classes)
        self.classifier_outcome = nn.Sequential(
            nn.Linear(flattened_size, 10 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, param.num_of_outcome) # Should be 2
        )

        # Classifier for CPC (5 classes)
        self.classifier_cpc = nn.Sequential(
            nn.Linear(flattened_size, 10 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, param.num_of_cpc)  # Should be 5
        )

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, channels, seq_len, patch_size)
           In your case: (batch_size, 18, 30, 200)
        """
        # x = x / 100
        bz, ch_num, seq_len, patch_size = x.shape
        # Pass through the backbone. The backbone is assumed to output 
        # a tensor of shape (batch_size, channels, seq_len, d_model)
        feats = self.backbone(x)
        # Flatten the features: (batch_size, channels*seq_len*d_model)
        out = feats.contiguous().view(bz, ch_num*seq_len*200)

        # Obtain the predictions from each classifier head.       
        out_outcome = self.classifier_outcome(out)
        out_cpc = self.classifier_cpc(out)
        
        return out_outcome, out_cpc
