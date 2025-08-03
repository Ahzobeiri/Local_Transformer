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
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from pretrained.LMDB_data_loader import LMDBChannelEpochDataset

# --- NEW: Imports for expanded feature extraction ---
from scipy.stats import skew, kurtosis
from scipy.signal import welch
# This new dependency is required for entropy and fractal dimension features.
# Install with: pip install antropy-eeg
import antropy as ant

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
    parser.add_argument('--window_size', default=15, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--temporal_context_modules', choices=['lstm','mha','lstm_mha','mamba'], default='mamba')
    parser.add_argument('--holdout_subject_size', default=50, type=int)
    parser.add_argument('--sfreq', default=200, type=int)
    parser.add_argument('--test_size', default=0.10, type=float)
    # Train Hyperparameter
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
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--projection_hidden', default=[1024, 512], type=list)
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--mask_ratio', default=0.8, type=float)
    parser.add_argument('--print_point', default=20, type=int)
    return parser.parse_args()

# --- MODIFIED: Comprehensive EEG Feature Extraction Function ---
def extract_eeg_features(data_batch, sfreq):
    """
    Extracts 21 time, frequency, and complexity features from a batch of EEG signals.
    Assumes signals are pre-bandpassed to [0.5, 40] Hz.
    Args:
        data_batch (torch.Tensor): A batch of 1D EEG signals, shape (batch_size, n_samples).
        sfreq (int): The sampling frequency.
    Returns:
        torch.Tensor: A tensor of features, shape (batch_size, 21).
    """
    data_np = data_batch.cpu().numpy()
    batch_size, n_samples = data_np.shape
    features = []

    # Adjusted delta band to match the [0.5, 40] Hz signal range.
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 40)}
    
    for i in range(batch_size):
        signal = data_np[i, :]
        
        # --- Time Domain Features (8 features) ---
        mean = np.mean(signal)
        std = np.std(signal)
        rms = np.sqrt(np.mean(signal**2))
        skewness = skew(signal)
        kurtosis_val = kurtosis(signal)
        zero_crossings = ant.num_zerocross(signal)
        
        # Hjorth Parameters
        activity = np.var(signal)
        mobility = ant.hjorth_params(signal)[0]
        complexity = ant.hjorth_params(signal)[1]
        
        # --- Frequency Domain Features (9 features) ---
        freqs, psd = welch(signal, sfreq=sfreq, nperseg=sfreq*2, noverlap=sfreq)
        
        # Absolute Band Powers
        band_powers = ant.bandpower(psd, freqs, bands=list(bands.values()), method='abs', relative=False)
        delta_p, theta_p, alpha_p, beta_p, gamma_p = band_powers
        
        # Band Ratios
        alpha_delta_ratio = alpha_p / (delta_p + 1e-8)
        alpha_theta_ratio = alpha_p / (theta_p + 1e-8)
        beta_alpha_theta_ratio = beta_p / (alpha_p + theta_p + 1e-8)

        # Spectral Slope
        idx_fit = np.logical_and(freqs >= 1, freqs <= 40)
        log_freqs = np.log10(freqs[idx_fit])
        log_psd = np.log10(psd[idx_fit])
        slope = np.polyfit(log_freqs, log_psd, 1)[0]
        
        # --- Complexity / Non-linear Features (4 features) ---
        spectral_entropy = ant.spectral_entropy(signal, sf=sfreq, method='welch', normalize=True)
        peak_freq = freqs[np.argmax(psd)]
        perm_entropy = ant.perm_entropy(signal, normalize=True)
        petrosian_fd = ant.petrosian_fd(signal)

        # Combine all 21 features for the current signal
        current_features = [
            mean, rms, std, skewness, kurtosis_val, zero_crossings, mobility, complexity,
            delta_p, theta_p, alpha_p, beta_p, gamma_p,
            alpha_delta_ratio, alpha_theta_ratio, beta_alpha_theta_ratio, slope,
            spectral_entropy, peak_freq, perm_entropy, petrosian_fd
        ]
        features.append(current_features)
        
    return torch.tensor(features, dtype=torch.float32).nan_to_num()


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
        return x_seq, ys[-1] # Label is from the last element in the sequence


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
        for param in backbone.parameters():
            param.requires_grad = False
        return backbone

class LSTM_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(embed_dim, 5) # Output size is 5 for CPC

    def forward(self, x):
        x = self.apply_backbone(x)
        x, _ = self.lstm(x)
        return self.fc(x)

class MHA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True), num_layers=2)
        self.fc = nn.Linear(embed_dim, 5) # Output size is 5

    def forward(self, x):
        x = self.apply_backbone(x)
        return self.fc(self.transformer(x))

class LSTM_MHA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=1, batch_first=True)
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True), num_layers=2)
        self.fc   = nn.Linear(embed_dim, 5) # Output size is 5

    def forward(self, x):
        x = self.apply_backbone(x)
        x, _ = self.lstm(x)
        x = self.trans(x)
        return self.fc(x)

# --- CORRECTED: MAMBA_TCM Class with fixed shape handling ---
class MAMBA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim, sfreq):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.sfreq = sfreq
        self.n_features = 21
        
        self.mamba = nn.Sequential(*[
            Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(1)
        ])
        
        self.fc_project = nn.Linear(embed_dim, 30) 
        self.fc_classify = nn.Linear(30 + self.n_features, 5)

    def forward(self, x):
        # x shape: (batch, seq_len, signal_length), e.g., (64, 15, 6000)
        
        # --- Step 1: Get sequence embeddings from Mamba ---
        x_embedded = self.apply_backbone(x)
        mamba_out = self.mamba(x_embedded) # Shape: (batch, seq_len, embed_dim)
        
        # --- Step 2: Extract handcrafted EEG features from the last time step ---
        # ### FIXED ###: Correctly select the last time snippet and pass it to the
        # feature extractor. The original code had a shape mismatch that would
        # cause a runtime error by incorrectly averaging the signal.
        last_snippet = x[:, -1, :] # Shape: (batch, signal_length)
        eeg_features = extract_eeg_features(last_snippet, self.sfreq).to(x.device) # Shape: (batch, 21)
        
        # --- Step 3: Combine features and classify ---
        last_mamba_out = mamba_out[:, -1, :] # Shape: (batch, embed_dim)
        projected_mamba_out = self.fc_project(last_mamba_out) # Shape: (batch, 30)
        combined_features = torch.cat((projected_mamba_out, eeg_features), dim=1) # Shape: (batch, 51)
        final_logits = self.fc_classify(combined_features) # Shape: (batch, 5)
        
        # --- Step 4: Format output to match other models ---
        output = torch.zeros(mamba_out.shape[0], mamba_out.shape[1], 5, device=x.device)
        output[:, -1, :] = final_logits
        
        return output


class Trainer:
    def __init__(self, args):
        self.args = args
        self.n_classes = 5
        # ─── Create NeuroNet from scratch ─────────────────────────────
        model_kwargs = {
            'fs':               args.sfreq, 'second':               args.second,
            'time_window':      args.time_window, 'time_step':        args.time_step,
            'encoder_embed_dim':  args.encoder_embed_dim, 'encoder_heads':      args.encoder_heads,
            'encoder_depths':     args.encoder_depths, 'decoder_embed_dim':  args.decoder_embed_dim,
            'decoder_heads':      args.decoder_heads, 'decoder_depths':     args.decoder_depths,
            'projection_hidden':  args.projection_hidden, 'temperature':        args.temperature
        }
        pretrained = NeuroNet(**model_kwargs)
        backbone = NeuroNetEncoderWrapper(
            fs=pretrained.fs, second=pretrained.second, time_window=pretrained.time_window,
            time_step=pretrained.time_step, frame_backbone=pretrained.frame_backbone,
            patch_embed=pretrained.autoencoder.patch_embed, encoder_block=pretrained.autoencoder.encoder_block,
            encoder_norm=pretrained.autoencoder.encoder_norm, cls_token=pretrained.autoencoder.cls_token,
            pos_embed=pretrained.autoencoder.pos_embed, final_length=pretrained.autoencoder.embed_dim
        )
        tcm_cls  = {
            'lstm': LSTM_TCM, 'mha': MHA_TCM,
            'lstm_mha': LSTM_MHA_TCM, 'mamba': MAMBA_TCM
        }[args.temporal_context_modules]

        # Conditionally pass sfreq to MAMBA_TCM
        if args.temporal_context_modules == 'mamba':
            self.model = tcm_cls(
                backbone, pretrained.autoencoder.embed_dim, args.embed_dim, sfreq=args.sfreq
            ).to(device)
        else:
            self.model = tcm_cls(
                backbone, pretrained.autoencoder.embed_dim, args.embed_dim
            ).to(device)

        self.tcm = self.model
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
        else: # Handle case where only one class is predicted
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
                out = self.tcm(x) # Shape: (batch, seq, 5)
                
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
        ds_train = LMDBChannelEpochDataset(self.args.base_path, 'train', fs=self.model.backbone.fs, n_channels=len(self.args.ch_names))
        ds_val   = LMDBChannelEpochDataset(self.args.base_path, 'val', fs=self.model.backbone.fs, n_channels=len(self.args.ch_names))
        ds_test  = LMDBChannelEpochDataset(self.args.base_path, 'test', fs=self.model.backbone.fs, n_channels=len(self.args.ch_names))
        
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
