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
# REMOVED: NeuroNet and NeuroNetEncoderWrapper imports are no longer needed.
# from models.utils import model_size
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)
from sklearn.preprocessing import label_binarize
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
    parser.add_argument('--ch_names', nargs='+', default=['Fp1','F7','T3','T5','O1',
                                                         'Fp2','F8','T4','T6','O2',
                                                         'F3','C3','P3','F4','C4',
                                                         'P4','Fz','Cz','Pz'])
    parser.add_argument('--temporal_context_length', default=15, type=int)
    parser.add_argument('--window_size', default=15, type=int) # Note: This is used as the step size for sequences
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--temporal_context_modules', choices=['lstm','mha','lstm_mha','mamba'], default='mamba')
    parser.add_argument('--holdout_subject_size', default=50, type=int)
    parser.add_argument('--sfreq', default=200, type=int)
    parser.add_argument('--test_size', default=0.10, type=float)
    
    # MODIFIED: Kept 'second' as it defines the input snippet duration.
    # Removed NeuroNet-specific hyperparameters.
    parser.add_argument('--second', default=30, type=int, help="Duration of each EEG snippet in seconds.")
    
    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=20, type=int)
    parser.add_argument('--train_base_learning_rate', default=1e-4, type=float)
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--train_batch_accumulation', default=1, type=int)
    
    # Model Hyperparameter
    parser.add_argument('--print_point', default=20, type=int)
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
        return x_seq, ys[-1] # Label is from the last element in the sequence

# MODIFIED: The base class is simplified. It no longer uses a backbone.
# It now takes the raw input dimension and projects it to an embedding space.
class TemporalContextModule(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        # This layer now acts as the feature extractor, replacing the NeuroNet backbone.
        # It projects the flattened raw EEG snippet into the desired embedding dimension.
        self.embed_layer = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def apply_embedding(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        
        # Reshape to process all snippets in a batch at once
        x_reshaped = x.reshape(batch_size * seq_len, input_dim)
        
        # Apply the embedding layer
        embedded_x = self.embed_layer(x_reshaped)
        
        # Reshape back to sequence format
        output = embedded_x.reshape(batch_size, seq_len, -1)
        return output

# MODIFIED: All TCM subclasses now inherit from the new backbone-less TemporalContextModule.
class LSTM_TCM(TemporalContextModule):
    def __init__(self, input_dim, embed_dim):
        super().__init__(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(embed_dim, 5)

    def forward(self, x):
        x = self.apply_embedding(x) # Use the new embedding method
        x, _ = self.lstm(x)
        return self.fc(x)

class MHA_TCM(TemporalContextModule):
    def __init__(self, input_dim, embed_dim):
        super().__init__(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True), num_layers=2)
        self.fc = nn.Linear(embed_dim, 5)

    def forward(self, x):
        x = self.apply_embedding(x) # Use the new embedding method
        return self.fc(self.transformer(x))

class LSTM_MHA_TCM(TemporalContextModule):
    def __init__(self, input_dim, embed_dim):
        super().__init__(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=1, batch_first=True)
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True), num_layers=2)
        self.fc   = nn.Linear(embed_dim, 5)

    def forward(self, x):
        x = self.apply_embedding(x) # Use the new embedding method
        x, _ = self.lstm(x)
        x = self.trans(x)
        return self.fc(x)

class MAMBA_TCM(TemporalContextModule):
    def __init__(self, input_dim, embed_dim):
        super().__init__(input_dim, embed_dim)
        self.mamba = nn.Sequential(*[
            Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(1)
        ])
        self.fc = nn.Linear(embed_dim, 5)

    def forward(self, x):
        x = self.apply_embedding(x) # Use the new embedding method
        return self.fc(self.mamba(x))

class Trainer:
    def __init__(self, args):
        self.args = args
        self.n_classes = 5
        
        # MODIFIED: Removed all NeuroNet instantiation logic.
        # The model is now created directly without a separate backbone.
        
        # Calculate the input dimension based on the raw data shape.
        # Assumes each snippet is a flattened vector of: channels * sampling_rate * duration
        input_dim = len(args.ch_names) * args.sfreq * args.second
        
        tcm_cls  = {
            'lstm': LSTM_TCM, 'mha': MHA_TCM,
            'lstm_mha': LSTM_MHA_TCM, 'mamba': MAMBA_TCM
        }[args.temporal_context_modules]
        
        # Instantiate the model with the calculated input dimension.
        self.model = tcm_cls(input_dim, args.embed_dim).to(device)
        self.tcm = self.model
        
        print(f"Model created: {args.temporal_context_modules.upper()}")
        print(f"Input snippet dimension: {input_dim}")
        print(f"Embedding dimension: {args.embed_dim}")

        self.criterion = nn.CrossEntropyLoss()

    ###--- BINARY OUTCOME EVALUATION ---###
    def compute_metrics_binary(self, y_true, y_prob_pos_class, y_pred):
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average='binary', zero_division=0)
        try:
            auc_ = roc_auc_score(y_true, y_prob_pos_class)
        except ValueError:
            auc_ = float('nan')
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob_pos_class, pos_label=1)
        aupr = auc(recall, precision)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        
        return {
            'accuracy': acc, 'f1': f1, 'auc': auc_, 'aupr': aupr,
            'sensitivity': sens, 'specificity': spec, 'confusion_matrix': cm
        }

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

    ###--- CPC (5-CLASS) EVALUATION ---###
    def compute_metrics_cpc(self, y_true, y_prob, y_pred):
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        y_true_binarized = label_binarize(y_true, classes=range(self.n_classes))
        try:
            auc_ = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
        except ValueError:
            auc_ = float('nan')
        
        aupr_scores = []
        for i in range(self.n_classes):
            if np.sum(y_true_binarized[:, i]) > 0:
                precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_prob[:, i])
                aupr_scores.append(auc(recall, precision))
        aupr = np.mean(aupr_scores) if aupr_scores else float('nan')
        cm = confusion_matrix(y_true, y_pred, labels=range(self.n_classes))
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
        # MODIFIED: The dataset loader no longer depends on the model for the 'fs' parameter.
        # It now uses the 'sfreq' argument directly.
        ds_train = LMDBChannelEpochDataset(self.args.base_path, 'train', fs=self.args.sfreq, n_channels=len(self.args.ch_names))
        ds_val   = LMDBChannelEpochDataset(self.args.base_path, 'val', fs=self.args.sfreq, n_channels=len(self.args.ch_names))
        ds_test  = LMDBChannelEpochDataset(self.args.base_path, 'test', fs=self.args.sfreq, n_channels=len(self.args.ch_names))
        
        X_train_val = np.concatenate([ds_train.total_x, ds_val.total_x], axis=0)
        Y_train_val_cpc = np.concatenate([ds_train.total_y, ds_val.total_y], axis=0)
        
        train_val_snips = ArraySnippetDataset(X_train_val, Y_train_val_cpc)
        train_val_seq = LMDBSequenceDataset(train_val_snips, self.args.temporal_context_length, self.args.window_size)
        all_seq_labels = np.array([train_val_seq[i][1] for i in range(len(train_val_seq))], dtype=np.int64)
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        fold_metrics_binary, fold_metrics_cpc = [], []
        
        best_overall_f1 = 0
        best_overall_state = None
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_seq), 1):
            print(f"\n=== Fold {fold}/5 ===")
            train_sub, val_sub = Subset(train_val_seq, train_idx), Subset(train_val_seq, val_idx)
            
            train_labels_for_fold = all_seq_labels[train_idx]
            class_counts_fold = np.bincount(train_labels_for_fold, minlength=self.n_classes)
            weights_per_class = 1.0 / (class_counts_fold + 1e-6)
            sample_weights_fold = weights_per_class[train_labels_for_fold]
            
            sampler = WeightedRandomSampler(weights=sample_weights_fold, num_samples=len(sample_weights_fold), replacement=True)
            
            tr = DataLoader(train_sub, batch_size=self.args.batch_size, sampler=sampler, drop_last=True)
            vl = DataLoader(val_sub,   batch_size=self.args.batch_size, shuffle=False, drop_last=False)
            self.model.apply(lambda m: isinstance(m, nn.Linear) and hasattr(m, 'reset_parameters') and m.reset_parameters())
            optim = opt.AdamW(self.tcm.parameters(), lr=self.args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.args.epochs)
            best_fold_f1 = 0
            best_fold_state = None
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
                
                print(f"\n--- Fold {fold} Epoch {ep+1} Validation Metrics ---")
                
                metrics_binary = self.evaluate_loader_binary(vl)
                metrics_cpc = self.evaluate_loader_cpc(vl)
                
                print("  Binary Outcome:")
                for k, v in metrics_binary.items():
                    if k != 'confusion_matrix':
                        print(f"    {k}: {v:.4f}")
                print("  CPC (5-Class) Outcome:")
                for k, v in metrics_cpc.items():
                    if k != 'confusion_matrix':
                        print(f"    {k}: {v:.4f}")
                
                if metrics_cpc['f1'] > best_fold_f1:
                    best_fold_f1 = metrics_cpc['f1']
                    best_fold_state = self.tcm.state_dict()
                    print("    (New best model for this fold saved)")
                print("-" * 40)
            
            self.tcm.load_state_dict(best_fold_state)
            val_metrics_binary = self.evaluate_loader_binary(vl)
            val_metrics_cpc = self.evaluate_loader_cpc(vl)
            fold_metrics_binary.append(val_metrics_binary)
            fold_metrics_cpc.append(val_metrics_cpc)
            
            print("\n--- Best Validation Metrics for this Fold ---")
            print("  Binary Outcome:")
            for k,v in val_metrics_binary.items():
                if k == 'confusion_matrix':
                    print(f"    {k}:\n{v}")
                else:
                    print(f"    {k}: {v:.4f}")
            print("  CPC (5-Class) Outcome:")
            for k,v in val_metrics_cpc.items():
                if k == 'confusion_matrix':
                    print(f"    {k}:\n{v}")
                else:
                    print(f"    {k}: {v:.4f}")
            if best_fold_f1 > best_overall_f1:
                best_overall_f1 = best_fold_f1
                best_overall_state = best_fold_state
        
        print("\n" + "="*40)
        print("=== Mean CV Validation Metrics (Binary) ===")
        for k in fold_metrics_binary[0]:
            if k != 'confusion_matrix':
                mean_val = np.mean([m[k] for m in fold_metrics_binary])
                print(f"  Mean {k}: {mean_val:.4f}")
        print("\n=== Mean CV Validation Metrics (CPC) ===")
        for k in fold_metrics_cpc[0]:
            if k != 'confusion_matrix':
                mean_val = np.mean([m[k] for m in fold_metrics_cpc])
                print(f"  Mean {k}: {mean_val:.4f}")

        print("\n" + "="*40)
        print("=== Held-out Test Metrics ===")
        self.tcm.load_state_dict(best_overall_state)
        Y_test_cpc = ds_test.total_y
        test_snips = ArraySnippetDataset(ds_test.total_x, Y_test_cpc)
        test_seq = LMDBSequenceDataset(test_snips, self.args.temporal_context_length, self.args.window_size)
        test_loader = DataLoader(test_seq, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        
        test_metrics_binary = self.evaluate_loader_binary(test_loader)
        test_metrics_cpc = self.evaluate_loader_cpc(test_loader)
        
        print("\n--- Test Metrics (Binary Outcome) ---")
        for k,v in test_metrics_binary.items():
            if k == 'confusion_matrix':
                print(f"  {k}:\n{v}")
            else:
                print(f"  {k}: {v:.4f}")
        print("\n--- Test Metrics (CPC Outcome) ---")
        for k,v in test_metrics_cpc.items():
            if k == 'confusion_matrix':
                print(f"  {k}:\n{v}")
            else:
                print(f"  {k}: {v:.4f}")
        print("="*40)

if __name__ == '__main__':
    args = get_args()
    Trainer(args).train()
