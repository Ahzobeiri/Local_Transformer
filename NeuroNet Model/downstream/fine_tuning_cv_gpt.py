# -*- coding:utf-8 -*-
import os
import pickle
import lmdb
import torch
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as opt
from mamba_ssm import Mamba
from models.utils import model_size
from torch.utils.data import Dataset, DataLoader, Subset
from models.neuronet.model import NeuroNet, NeuroNetEncoderWrapper
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from pretrained.LMDB_data_loader import LMDBChannelEpochDataset

warnings.filterwarnings(action='ignore')

# Reproducibility
random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default='/projects/scratch/fhajati/ckpt/unimodal/eeg', type=str)
    parser.add_argument('--base_path', default='/projects/scratch/fhajati/physionet.org/files/LMDB_DATA/19Ch_Last1h', type=str)
    parser.add_argument('--ch_names', nargs='+', default=['Fp1','F7','T3','T5','O1',
                                                           'Fp2','F8','T4','T6','O2',
                                                           'F3','C3','P3','F4','C4',
                                                           'P4','Fz','Cz','Pz'])
    parser.add_argument('--temporal_context_length', default=20, type=int)
    parser.add_argument('--window_size', default=10, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--temporal_context_modules', choices=['lstm','mha','lstm_mha','mamba'], default='mamba')
    return parser.parse_args()


class ArraySnippetDataset(Dataset):
    """Wraps pre-loaded arrays into a PyTorch Dataset."""
    def __init__(self, x_arr: np.ndarray, y_arr: np.ndarray):
        self.x = torch.tensor(x_arr, dtype=torch.float32)
        self.y = torch.tensor(y_arr, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LMDBSequenceDataset(Dataset):
    """
    Wraps per-channel snippets into overlapping sequences of length seq_len.
    Returns x_seq: (seq_len, snippet_dim), y: scalar
    """
    def __init__(self, snippet_ds: Dataset, seq_len: int, step: int = None):
        self.snippet_ds = snippet_ds
        self.seq_len    = seq_len
        self.step       = step or seq_len
        total_snips = len(self.snippet_ds)
        self.indices = list(range(0, total_snips - seq_len + 1, self.step))
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        start = self.indices[idx]
        xs, ys = [], []
        for offset in range(self.seq_len):
            x, y = self.snippet_ds[start + offset]
            xs.append(x)
            ys.append(y)
        x_seq = torch.stack(xs, dim=0)
        return x_seq, ys[0]


class TemporalContextModule(nn.Module):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__()
        self.backbone = self.freeze_backbone(backbone)
        self.embed_layer = nn.Sequential(
            nn.Linear(backbone_final_length, embed_dim),
            nn.BatchNorm1d(embed_dim), nn.ELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def apply_backbone(self, x):
        out = []
        for x_ in torch.split(x, dim=1, split_size_or_sections=1):
            o = self.backbone(x_.squeeze())
            o = self.embed_layer(o)
            out.append(o)
        return torch.stack(out, dim=1)
    @staticmethod
    def freeze_backbone(backbone):
        for param in backbone.parameters():
            param.requires_grad = False
        return backbone

class LSTM_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=2)
        self.fc   = nn.Linear(embed_dim, 5)
    def forward(self, x):
        x = self.apply_backbone(x)
        x, _ = self.lstm(x)
        return self.fc(x)

class MHA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8), num_layers=2)
        self.fc = nn.Linear(embed_dim, 5)
    def forward(self, x):
        x = self.apply_backbone(x)
        return self.fc(self.transformer(x))

class LSTM_MHA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=1)
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8), num_layers=2)
        self.fc   = nn.Linear(embed_dim, 5)
    def forward(self, x):
        x = self.apply_backbone(x)
        x, _ = self.lstm(x)
        x = self.trans(x)
        return self.fc(x)

class MAMBA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.mamba = nn.Sequential(*[
            Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(1)
        ])
        self.fc = nn.Linear(embed_dim, 5)
    def forward(self, x):
        x = self.apply_backbone(x)
        return self.fc(self.mamba(x))


class Trainer:
    def __init__(self, args):
        self.args   = args
        ckpt       = torch.load(os.path.join(args.ckpt_path, 'model/best_model.pth'), map_location='cpu')
        param      = ckpt['model_parameter']
        pretrained = NeuroNet(**param)
        pretrained.load_state_dict(ckpt['model_state'])
        backbone   = NeuroNetEncoderWrapper(
            fs=param['fs'], second=param['second'],
            time_window=param['time_window'], time_step=param['time_step'],
            frame_backbone=pretrained.frame_backbone,
            patch_embed=pretrained.autoencoder.patch_embed,
            encoder_block=pretrained.autoencoder.encoder_block,
            encoder_norm=pretrained.autoencoder.encoder_norm,
            cls_token=pretrained.autoencoder.cls_token,
            pos_embed=pretrained.autoencoder.pos_embed,
            final_length=pretrained.autoencoder.embed_dim
        )
        tcm_cls   = {
            'lstm': LSTM_TCM, 'mha': MHA_TCM,
            'lstm_mha': LSTM_MHA_TCM, 'mamba': MAMBA_TCM
        }[args.temporal_context_modules]
        self.model = tcm_cls(backbone, pretrained.autoencoder.embed_dim, args.embed_dim).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        # --- load ALL snippets ---
        train_ds = LMDBChannelEpochDataset(self.args.base_path, mode='train', fs=self.model.backbone.fs,
                                           n_channels=len(self.args.ch_names))
        val_ds   = LMDBChannelEpochDataset(self.args.base_path, mode='val',   fs=self.model.backbone.fs,
                                           n_channels=len(self.args.ch_names))
        test_ds  = LMDBChannelEpochDataset(self.args.base_path, mode='test',  fs=self.model.backbone.fs,
                                           n_channels=len(self.args.ch_names))
        X = np.concatenate([train_ds.total_x, val_ds.total_x, test_ds.total_x], axis=0)
        Y = np.concatenate([train_ds.total_y, val_ds.total_y, test_ds.total_y], axis=0)
        all_snips = ArraySnippetDataset(X, Y)

        # --- wrap sequences ---
        full_seq_ds = LMDBSequenceDataset(all_snips,
            seq_len=self.args.temporal_context_length,
            step=self.args.window_size)

        # --- 5‑fold CV ---
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(full_seq_ds), 1):
            print(f"\n### Fold {fold}/5 ###")
            train_sub = Subset(full_seq_ds, train_idx)
            val_sub   = Subset(full_seq_ds, val_idx)
            train_loader = DataLoader(train_sub, batch_size=self.args.batch_size, shuffle=True,  drop_last=True)
            val_loader   = DataLoader(val_sub,   batch_size=self.args.batch_size, shuffle=False, drop_last=True)

            optimizer = opt.AdamW(self.model.parameters(), lr=self.args.lr)
            scheduler = opt.CosineAnnealingLR(optimizer, T_max=self.args.epochs)

            best_mf1 = 0.0
            for epoch in range(self.args.epochs):
                # train
                self.model.train()
                for x, y in tqdm(train_loader, desc=f"Fold{fold} Train E{epoch+1}", leave=False):
                    optimizer.zero_grad()
                    x, y = x.to(device), y.to(device)
                    out = self.model(x)
                    pred_last = out[:, -1, :]
                    loss = self.criterion(pred_last, y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                # val
                self.model.eval()
                all_pred, all_true = [], []
                with torch.no_grad():
                    for x, y in tqdm(val_loader, desc=f"Fold{fold} Val E{epoch+1}", leave=False):
                        x, y = x.to(device), y.to(device)
                        out = self.model(x)
                        pred = out[:, -1, :].argmax(dim=-1)
                        all_pred.extend(pred.cpu().numpy())
                        all_true.extend(y.cpu().numpy())
                mf1 = f1_score(all_true, all_pred, average='macro')
                if mf1 > best_mf1:
                    best_mf1 = mf1
            fold_scores.append(best_mf1)
            print(f"Fold {fold} best Macro‑F1 = {best_mf1:.4f}")

        print("\n5‑Fold CV Macro‑F1 scores:", fold_scores)
        print("Mean Macro‑F1:", np.mean(fold_scores))


if __name__ == '__main__':
    args = get_args()
    Trainer(args).train()
