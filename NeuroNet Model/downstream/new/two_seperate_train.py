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
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from models.neuronet.model import NeuroNet, NeuroNetEncoderWrapper
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from pretrained.LMDB_data_loader import LMDBChannelEpochDataset

warnings.filterwarnings(action='ignore')

# reproducibility
random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default='/projects/scratch/fhajati/ckpt/unimodal/eeg', type=str)
    parser.add_argument('--base_path', default='/projects/scratch/fhajati/physionet.org/files/LMDB_DATA/19Ch_Last1h', type=str)
    parser.add_argument('--temporal_context_length', default=20, type=int)
    parser.add_argument('--window_size', default=10, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--temporal_context_modules', choices=['lstm','mha','lstm_mha','mamba'], default='mamba')
    parser.add_argument('--ch_names', nargs='+', default=['Fp1','F7','T3','T5','O1',
                                                           'Fp2','F8','T4','T6','O2',
                                                           'F3','C3','P3','F4','C4',
                                                           'P4','Fz','Cz','Pz'])
    parser.add_argument('--holdout_subject_size', default=50, type=int)
    parser.add_argument('--sfreq', default=100, type=int)
    parser.add_argument('--test_size', default=0.10, type=float)
    
    # Train Hyperparameter
    parser.add_argument('--pretrain_epochs', default=5, type=int, help="Number of epochs to pre-train the NeuroNet backbone.")
    parser.add_argument('--train_epochs', default=20, type=int)
    parser.add_argument('--train_base_learning_rate', default=1e-4, type=float)
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--train_batch_accumulation', default=1, type=int)
    
    # Model Hyperparameter
    parser.add_argument('--second', default=30, type=int)
    parser.add_argument('--time_window', default=4, type=int)
    parser.add_argument('--time_step', default=1, type=int)
    parser.add_argument('--encoder_embed_dim', default=768, type=int)
    parser.add_argument('--encoder_heads', default=8, type=int)
    parser.add_argument('--encoder_depths', default=4, type=int)
    parser.add_argument('--decoder_embed_dim', default=256, type=int)
    parser.add_argument('--decoder_heads', default=8, type=int)
    parser.add_argument('--decoder_depths', default=3, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--projection_hidden', default=[1024, 512], type=list)
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--mask_ratio', default=0.8, type=float)
    parser.add_argument('--print_point', default=20, type=int)
    return parser.parse_args()


class ArraySnippetDataset(Dataset):
    """Wraps pre‐loaded arrays into a PyTorch Dataset."""
    def __init__(self, x_arr: np.ndarray, y_arr: np.ndarray):
        self.x = torch.tensor(x_arr, dtype=torch.float32)
        self.y = torch.tensor(y_arr, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LMDBSequenceDataset(Dataset):
    """
    Wraps per‐channel snippets into overlapping sequences of length seq_len.
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
        self.args = args
        
        # ─── Create NeuroNet from scratch ──────────────────────────
        # We will pre-train this model before building the final TCM.
        model_kwargs = {
            'fs':                 args.sfreq,
            'second':             args.second,
            'time_window':        args.time_window,
            'time_step':          args.time_step,
            'encoder_embed_dim':  args.encoder_embed_dim,
            'encoder_heads':      args.encoder_heads,
            'encoder_depths':     args.encoder_depths,
            'decoder_embed_dim':  args.decoder_embed_dim,
            'decoder_heads':      args.decoder_heads,
            'decoder_depths':     args.decoder_depths,
            'projection_hidden':  args.projection_hidden,
            'temperature':        args.temperature
        }
        self.neuronet = NeuroNet(**model_kwargs).to(device)

        # ─── Compute class‐weights from the combined dataset ───────────
        # We need class frequencies before we instantiate the DataLoaders.
        ds_train = LMDBChannelEpochDataset(self.args.base_path, 'train',
                                           fs=args.sfreq, n_channels=len(self.args.ch_names))
        ds_val   = LMDBChannelEpochDataset(self.args.base_path, 'val',
                                           fs=args.sfreq, n_channels=len(self.args.ch_names))
        ds_test  = LMDBChannelEpochDataset(self.args.base_path, 'test',
                                           fs=args.sfreq, n_channels=len(self.args.ch_names))
        all_y = np.concatenate([ds_train.total_y, ds_val.total_y, ds_test.total_y])
        class_counts = np.bincount(all_y, minlength=5).astype(np.float32)
        # inverse frequency
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * 5.0
        self.class_weights = torch.tensor(class_weights, device=device)
        
        # The main model (TCM) and its criterion will be defined in the train() method
        # after NeuroNet has been pre-trained.
        self.model = None
        self.tcm = None
        self.criterion = None


    def compute_metrics(self, y_true, y_prob, y_pred):
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average='macro')
        try:
            auc_ = roc_auc_score(y_true, y_prob, multi_class='ovo')
        except:
            auc_ = float('nan')
        # AUPR
        precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
        aupr = auc(recall, precision)
        cm = confusion_matrix(y_true, y_pred)
        # per-class sensitivity and specificity
        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        tn = cm.sum() - (tp + fp + fn)
        sens = np.mean(tp / (tp + fn + 1e-8))
        spec = np.mean(tn / (tn + fp + 1e-8))
        return {
            'accuracy':acc,'f1':f1,'auc':auc_,'aupr':aupr,
            'sensitivity':sens,'specificity':spec,'confusion_matrix':cm
        }

    def evaluate_loader(self, loader):
        self.tcm.eval()
        all_true, all_pred, all_prob = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = self.tcm(x)
                probs = torch.softmax(out[:, -1, :], dim=-1)
                pred  = probs.argmax(dim=-1)
                # probability of the true class
                probs = probs.cpu().numpy()
                idx   = y.cpu().numpy()
                prob_true = probs[np.arange(len(idx)), idx]
                all_true.extend(idx)
                all_pred.extend(pred.cpu().numpy())
                all_prob.extend(prob_true)
        return self.compute_metrics(all_true, all_prob, all_pred)

    def train(self):
        # load all 3 splits, then pool together
        ds_train = LMDBChannelEpochDataset(self.args.base_path,'train',fs=self.args.sfreq,n_channels=len(self.args.ch_names))
        ds_val   = LMDBChannelEpochDataset(self.args.base_path,'val',fs=self.args.sfreq,  n_channels=len(self.args.ch_names))
        ds_test  = LMDBChannelEpochDataset(self.args.base_path,'test',fs=self.args.sfreq, n_channels=len(self.args.ch_names))
        X = np.concatenate([ds_train.total_x, ds_val.total_x, ds_test.total_x], axis=0)
        Y = np.concatenate([ds_train.total_y, ds_val.total_y, ds_test.total_y], axis=0)
        full_snips = ArraySnippetDataset(X, Y)

        # ─── Pre-train NeuroNet as an autoencoder ──────────────────────
        print(f"\n=== Pre-training NeuroNet for {self.args.pretrain_epochs} epochs ===")
        pretrain_loader = DataLoader(full_snips, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True)
        pretrain_optim = opt.AdamW(self.neuronet.parameters(), lr=self.args.train_base_learning_rate)

        self.neuronet.train()
        for ep in range(self.args.pretrain_epochs):
            total_loss = 0
            pbar = tqdm(pretrain_loader, desc=f"Pre-train Ep{ep+1}", leave=False)
            for x, _ in pbar: # We don't need labels 'y' for self-supervised pre-training
                x = x.to(device)
                pretrain_optim.zero_grad()
                
                # --- FIX IS HERE ---
                # Calculate loss as done in the NeuroNet pre-training script.
                # NeuroNet's forward pass returns (recon_loss, contrastive_loss, ...).
                recon_loss, contrastive_loss, _ = self.neuronet(x, mask_ratio=self.args.mask_ratio)
                loss = recon_loss + self.args.alpha * contrastive_loss
                # --- END OF FIX ---

                loss.backward()
                pretrain_optim.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            avg_loss = total_loss / len(pretrain_loader)
            print(f"Pre-train Epoch {ep+1}/{self.args.pretrain_epochs}, Average Pre-train Loss: {avg_loss:.4f}")
        print("=== Pre-training complete. Building TCM with pre-trained backbone. ===")
        # ─── END OF PRE-TRAINING ────────────────────────────────────────────

        # Now, build the main model using the pre-trained NeuroNet
        backbone = NeuroNetEncoderWrapper(
          fs=self.neuronet.fs,
          second=self.neuronet.second,
          time_window=self.neuronet.time_window,
          time_step=self.neuronet.time_step,
          frame_backbone=self.neuronet.frame_backbone,
          patch_embed=self.neuronet.autoencoder.patch_embed,
          encoder_block=self.neuronet.autoencoder.encoder_block,
          encoder_norm=self.neuronet.autoencoder.encoder_norm,
          cls_token=self.neuronet.autoencoder.cls_token,
          pos_embed=self.neuronet.autoencoder.pos_embed,
          final_length=self.neuronet.autoencoder.embed_dim
        )
        tcm_cls   = {
            'lstm': LSTM_TCM, 'mha': MHA_TCM,
            'lstm_mha': LSTM_MHA_TCM, 'mamba': MAMBA_TCM
        }[self.args.temporal_context_modules]

        self.model = tcm_cls(backbone, self.neuronet.autoencoder.embed_dim, self.args.embed_dim).to(device)
        self.tcm = self.model # for compatibility with existing evaluation code
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # Prepare for main training
        full_seq = LMDBSequenceDataset(full_snips, self.args.temporal_context_length, self.args.window_size)
        
        # build a sampler to upsample minority classes
        labels = []
        for idx in range(len(full_seq)):
            _, lbl = full_seq[idx]
            labels.append(int(lbl))
        labels = np.array(labels)
        
        all_y_seq = np.concatenate([ds_train.total_y, ds_val.total_y, ds_test.total_y])
        class_counts_seq = np.bincount(all_y_seq, minlength=5).astype(np.float32)
        sample_weights = 1.0 / (class_counts_seq[labels] + 1e-6)
        sample_weights = torch.from_numpy(sample_weights)

        # 5‑fold CV on full_seq
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(full_seq), 1):
            print(f"\n=== Fold {fold}/5 ===")
            train_sub = Subset(full_seq, train_idx)
            val_sub   = Subset(full_seq, val_idx)
          
            train_sampler = WeightedRandomSampler(
              weights=sample_weights[train_idx],
              num_samples=len(train_idx),
              replacement=True)
            
            tr = DataLoader(train_sub, batch_size=self.args.batch_size, sampler=train_sampler, drop_last=True)
            vl = DataLoader(val_sub,   batch_size=self.args.batch_size, shuffle=False, drop_last=False)
            
            # Note: The backbone is frozen, so the optimizer only trains the new layers in TCM.
            optim = opt.AdamW(filter(lambda p: p.requires_grad, self.tcm.parameters()), lr=self.args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.args.epochs)
            best_f1 = 0
            best_state = None
            for ep in range(self.args.epochs):
                self.tcm.train()
                for x,y in tqdm(tr, desc=f"Fold{fold} Ep{ep+1}", leave=False):
                    x, y = x.to(device), y.to(device)
                    optim.zero_grad()
                    out = self.tcm(x)
                    loss = self.criterion(out[:, -1, :], y)
                    loss.backward()
                    optim.step()
                scheduler.step()
                metrics = self.evaluate_loader(vl)
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_state = self.tcm.state_dict()
            # restore best
            if best_state:
                self.tcm.load_state_dict(best_state)
            
            print(f"Fold {fold} Validation metrics:")
            fold_val_metrics = self.evaluate_loader(vl)
            for k,v in fold_val_metrics.items():
                if k != 'confusion_matrix':
                    print(f"  {k}: {v:.4f}")
            fold_metrics.append(fold_val_metrics)

        # report mean validation
        print("\n=== Mean CV Validation Metrics ===")
        mean_metrics = {}
        for k in fold_metrics[0].keys():
            if k != 'confusion_matrix':
                mean_metrics[k] = np.mean([m[k] for m in fold_metrics if k in m and not np.isnan(m[k])])
        for k,v in mean_metrics.items():
            print(f"  {k}: {v:.4f}")

        # finally evaluate on held‑out test split
        print("\n=== Held‑out Test Metrics ===")
        test_seq = LMDBSequenceDataset(ds_test, self.args.temporal_context_length, self.args.window_size)
        test_loader = DataLoader(test_seq, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        test_metrics = self.evaluate_loader(test_loader)
        for k,v in test_metrics.items():
            if k != 'confusion_matrix':
                print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    args = get_args()
    Trainer(args).train()
