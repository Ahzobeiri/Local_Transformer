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
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# KEY CHANGE: Import only the main data loader from the separate file
from data_loader import LMDBChannelEpochDataset

# KEY CHANGE: Define the pipeline-specific helper Dataset classes here
class ArraySnippetDataset(Dataset):
    """Wraps pre-loaded numpy arrays into a PyTorch Dataset."""
    def __init__(self, x_arr: np.ndarray, y_arr: np.ndarray):
        self.x = torch.tensor(x_arr, dtype=torch.float32)
        self.y = torch.tensor(y_arr, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LMDBSequenceDataset(Dataset):
    """
    Wraps individual epochs (snippets) into overlapping sequences
    for the temporal context model.
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
        return x_seq, ys[-1]

# Placeholder classes for NeuroNet (replace with your actual model imports)
class NeuroNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fs, self.second = kwargs.get('fs', 200), kwargs.get('second', 30)
        self.time_window, self.time_step = kwargs.get('time_window', 4), kwargs.get('time_step', 1)
        embed_dim = kwargs.get('encoder_embed_dim', 768)
        self.frame_backbone = nn.Identity()
        self.autoencoder = nn.Module()
        self.autoencoder.patch_embed = nn.Identity()
        self.autoencoder.encoder_block = nn.Identity()
        self.autoencoder.encoder_norm = nn.Identity()
        self.autoencoder.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.autoencoder.pos_embed = nn.Parameter(torch.zeros(1, 100, embed_dim))
        self.autoencoder.embed_dim = embed_dim

class NeuroNetEncoderWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.final_length = kwargs.get('final_length', 768)
    def forward(self, x):
        return torch.randn(x.shape[0], self.final_length)

warnings.filterwarnings(action='ignore')
random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='/path/to/your/LMDB_DATA', type=str)
    parser.add_argument('--ckpt_path', default='./ckpt', type=str)
    parser.add_argument('--ch_names', nargs='+', default=['Fp1','F7','T3','T5','O1','Fp2','F8','T4','T6','O2','F3','C3','P3','F4','C4','P4','Fz','Cz','Pz'])
    parser.add_argument('--temporal_context_length', default=15, type=int)
    parser.add_argument('--window_size', default=15, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--sfreq', default=200, type=int)
    parser.add_argument('--second', default=30, type=int)
    parser.add_argument('--time_window', default=4, type=int)
    parser.add_argument('--time_step', default=1, type=int)
    parser.add_argument('--encoder_embed_dim', default=768, type=int)
    return parser.parse_args()

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
            o = self.backbone(x_.squeeze(1))
            o = self.embed_layer(o)
            out.append(o)
        return torch.stack(out, dim=1)
    @staticmethod
    def freeze_backbone(backbone):
        for param in backbone.parameters(): param.requires_grad = False
        return backbone

class MAMBA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.mamba = nn.Sequential(*[Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2) for _ in range(1)])
        self.fc = nn.Linear(embed_dim, 5)
    def forward(self, x):
        x = self.apply_backbone(x)
        return self.fc(self.mamba(x))

class Trainer:
    def __init__(self, args):
        self.args = args
        self.n_classes = 5
        model_kwargs = {'fs': args.sfreq, 'second': args.second, 'time_window': args.time_window, 'time_step': args.time_step, 'encoder_embed_dim': args.encoder_embed_dim}
        pretrained = NeuroNet(**model_kwargs)
        backbone = NeuroNetEncoderWrapper(fs=pretrained.fs, second=pretrained.second, time_window=pretrained.time_window, time_step=pretrained.time_step, final_length=pretrained.autoencoder.embed_dim)
        self.model = MAMBA_TCM(backbone, pretrained.autoencoder.embed_dim, args.embed_dim).to(device)
        self.tcm = self.model
        self.criterion = nn.CrossEntropyLoss()

    def compute_metrics_binary(self, y_true, y_prob_pos_class, y_pred):
        acc, f1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='binary', zero_division=0)
        try: auc_ = roc_auc_score(y_true, y_prob_pos_class)
        except ValueError: auc_ = float('nan')
        precision, recall, _ = precision_recall_curve(y_true, y_prob_pos_class, pos_label=1)
        aupr = auc(recall, precision)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0,0,0,0))
        sens, spec = tp / (tp + fn + 1e-8), tn / (tn + fp + 1e-8)
        return {'accuracy': acc, 'f1': f1, 'auc': auc_, 'aupr': aupr, 'sensitivity': sens, 'specificity': spec, 'confusion_matrix': cm}

    def evaluate_loader_binary(self, loader):
        self.tcm.eval()
        all_true_binary, all_pred_binary, all_prob_bad_outcome = [], [], []
        with torch.no_grad():
            for x, y_cpc in loader:
                x, y_cpc = x.to(device), y_cpc.to(device)
                out = self.tcm(x)
                probs_cpc = torch.softmax(out[:, -1, :], dim=-1)
                y_true_binary = (y_cpc >= 3).long()
                prob_bad_outcome = probs_cpc[:, 3] + probs_cpc[:, 4]
                y_pred_binary = (prob_bad_outcome > 0.5).long()
                all_true_binary.extend(y_true_binary.cpu().numpy())
                all_pred_binary.extend(y_pred_binary.cpu().numpy())
                all_prob_bad_outcome.extend(prob_bad_outcome.cpu().numpy())
        return self.compute_metrics_binary(np.array(all_true_binary), np.array(all_prob_bad_outcome), np.array(all_pred_binary))

    def compute_metrics_cpc(self, y_true, y_prob, y_pred):
        acc, f1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro', zero_division=0)
        y_true_binarized = label_binarize(y_true, classes=range(self.n_classes))
        try: auc_ = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
        except ValueError: auc_ = float('nan')
        aupr_scores = [auc(recall, precision) for i in range(self.n_classes) if np.sum(y_true_binarized[:, i]) > 0 for precision, recall, _ in [precision_recall_curve(y_true_binarized[:, i], y_prob[:, i])]]
        aupr = np.mean(aupr_scores) if aupr_scores else float('nan')
        cm = confusion_matrix(y_true, y_pred, labels=range(self.n_classes))
        tp, fn, fp = np.diag(cm), cm.sum(axis=1) - np.diag(cm), cm.sum(axis=0) - np.diag(cm)
        tn = cm.sum() - (tp + fn + fp)
        sens, spec = np.mean(tp / (tp + fn + 1e-8)), np.mean(tn / (tn + fp + 1e-8))
        return {'accuracy':acc,'f1':f1,'auc':auc_,'aupr':aupr, 'sensitivity':sens,'specificity':spec,'confusion_matrix':cm}
        
    def evaluate_loader_cpc(self, loader):
        self.tcm.eval()
        all_true_cpc, all_pred_cpc, all_prob_cpc = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = self.tcm(x)
                probs = torch.softmax(out[:, -1, :], dim=-1)
                pred  = probs.argmax(dim=-1)
                all_true_cpc.extend(y.cpu().numpy())
                all_pred_cpc.extend(pred.cpu().numpy())
                all_prob_cpc.extend(probs.cpu().numpy())
        return self.compute_metrics_cpc(np.array(all_true_cpc), np.array(all_prob_cpc), np.array(all_pred_cpc))

    def train(self):
        print("Loading train, validation, and test datasets...")
        ds_train = LMDBChannelEpochDataset(self.args.base_path, 'train', fs=self.args.sfreq, n_channels=len(self.args.ch_names))
        ds_val   = LMDBChannelEpochDataset(self.args.base_path, 'val', fs=self.args.sfreq, n_channels=len(self.args.ch_names))
        ds_test  = LMDBChannelEpochDataset(self.args.base_path, 'test', fs=self.args.sfreq, n_channels=len(self.args.ch_names))
        
        print("Combining all datasets for subject-wise cross-validation...")
        X_all = np.concatenate([ds_train.total_x, ds_val.total_x, ds_test.total_x], axis=0)
        Y_all_cpc = np.concatenate([ds_train.total_y, ds_val.total_y, ds_test.total_y], axis=0)
        S_all = np.concatenate([ds_train.total_sids, ds_val.total_sids, ds_test.total_sids], axis=0)
        H_all = np.concatenate([ds_train.total_hospitals, ds_val.total_hospitals, ds_test.total_hospitals], axis=0)

        all_snips_ds = ArraySnippetDataset(X_all, Y_all_cpc)
        all_seq_ds = LMDBSequenceDataset(all_snips_ds, self.args.temporal_context_length, self.args.window_size)
        
        print("Generating subject groups for each sequence...")
        sequence_subject_ids = np.array([
            S_all[all_seq_ds.indices[i] + all_seq_ds.seq_len - 1] 
            for i in tqdm(range(len(all_seq_ds)), desc="Mapping sequences to subjects")
        ])
        gkf = GroupKFold(n_splits=5)
        fold_metrics_binary, fold_metrics_cpc = [], []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(np.arange(len(all_seq_ds)), groups=sequence_subject_ids), 1):
            print(f"\n=== Fold {fold}/5 ===")
            train_sub, test_sub = Subset(all_seq_ds, train_idx), Subset(all_seq_ds, test_idx)
            
            all_seq_labels = np.array([all_seq_ds[i][1] for i in range(len(all_seq_ds))], dtype=np.int64)
            train_labels_for_fold = all_seq_labels[train_idx]
            class_counts_fold = np.bincount(train_labels_for_fold, minlength=self.n_classes)
            weights_per_class = 1.0 / (class_counts_fold + 1e-6)
            sample_weights_fold = weights_per_class[train_labels_for_fold]
            sampler = WeightedRandomSampler(weights=sample_weights_fold, num_samples=len(sample_weights_fold), replacement=True)
            
            tr = DataLoader(train_sub, batch_size=self.args.batch_size, sampler=sampler, drop_last=True)
            ts = DataLoader(test_sub, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
            
            print("Re-initializing model weights for new fold...")
            self.model.apply(lambda m: isinstance(m, (nn.Linear, nn.LSTM, Mamba)) and hasattr(m, 'reset_parameters') and m.reset_parameters())
            optim = opt.AdamW(self.tcm.parameters(), lr=self.args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.args.epochs)
            best_fold_f1 = 0
            
            for ep in range(self.args.epochs):
                self.tcm.train()
                for x, y in tqdm(tr, desc=f"Fold {fold} Ep {ep+1}", leave=False):
                    x, y = x.to(device), y.to(device)
                    optim.zero_grad()
                    out = self.tcm(x)
                    loss = self.criterion(out[:, -1, :], y)
                    loss.backward()
                    optim.step()
                scheduler.step()
                
                metrics_cpc = self.evaluate_loader_cpc(ts)
                if metrics_cpc['f1'] > best_fold_f1:
                    best_fold_f1 = metrics_cpc['f1']
                    print(f"  (Epoch {ep+1}: New best F1: {best_fold_f1:.4f})")

            print(f"\n--- Final Test Metrics for Fold {fold} ---")
            fold_test_metrics_binary = self.evaluate_loader_binary(ts)
            fold_test_metrics_cpc = self.evaluate_loader_cpc(ts)
            fold_metrics_binary.append(fold_test_metrics_binary)
            fold_metrics_cpc.append(fold_test_metrics_cpc)
            
            print("  Binary Outcome:")
            for k,v in fold_test_metrics_binary.items():
                if k == 'confusion_matrix': print(f"    {k}:\n{v}")
                else: print(f"    {k}: {v:.4f}")
            
            print("  CPC (5-Class) Outcome:")
            for k,v in fold_test_metrics_cpc.items():
                if k == 'confusion_matrix': print(f"    {k}:\n{v}")
                else: print(f"    {k}: {v:.4f}")

        print("\n" + "="*50 + "\n=== Final Cross-Validation Results (5 Folds) ===\n" + "="*50)
        print("\n=== Mean & Std Dev CV Test Metrics (Binary) ===")
        for k in fold_metrics_binary[0]:
            if k != 'confusion_matrix':
                vals = [m[k] for m in fold_metrics_binary if not np.isnan(m[k])]
                print(f"  Mean {k}: {np.mean(vals):.4f} (±{np.std(vals):.4f})")

        print("\n=== Mean & Std Dev CV Test Metrics (CPC) ===")
        for k in fold_metrics_cpc[0]:
            if k != 'confusion_matrix':
                vals = [m[k] for m in fold_metrics_cpc if not np.isnan(m[k])]
                print(f"  Mean {k}: {np.mean(vals):.4f} (±{np.std(vals):.4f})")
        print("\n" + "="*50)

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    Trainer(args).train()
