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
from models.neuronet.model_CNN import NeuroNet, NeuroNetEncoderWrapper
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from pretrained.data_loader_channels import LMDBChannelEpochDataset

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

class MAMBA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.mamba = nn.Sequential(*[
            Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(1)
        ])
        self.fc = nn.Linear(embed_dim, 5) # Output size is 5

    def forward(self, x):
        x = self.apply_backbone(x)
        return self.fc(self.mamba(x))


class Trainer:
    def __init__(self, args):
        self.args = args
        self.n_classes = 5
        # ─── Create NeuroNet from scratch ─────────────────────────────
        model_kwargs = {
            'fs':                 args.sfreq, 'second':             args.second,
            'time_window':        args.time_window, 'time_step':          args.time_step,
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
        self.model = tcm_cls(backbone, pretrained.autoencoder.embed_dim, args.embed_dim).to(device)
        self.tcm = self.model
        
        # MODIFICATION 1: Use unweighted loss. The sampler will handle imbalance.
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
                
                # Get 5-class probabilities
                probs_cpc = torch.softmax(out[:, -1, :], dim=-1)
                
                # Convert to binary task
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
        
        # AUC and AUPR
        y_true_binarized = label_binarize(y_true, classes=range(self.n_classes))
        try:
            # OvO AUC requires probabilities for each class
            auc_ = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
        except ValueError:
            auc_ = float('nan')
        
        # Macro AUPR
        aupr_scores = []
        for i in range(self.n_classes):
            if np.sum(y_true_binarized[:, i]) > 0: # Check if class is present
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
        # --- STEP 1: LOAD AND PARTITION DATA CORRECTLY ---
        # Load all three data splits. The test set (ds_test) will be kept
        # completely separate until the final evaluation.
        print("--- Loading Data Partitions ---")
        ds_train = LMDBChannelEpochDataset(self.args.base_path, 'train', fs=self.model.backbone.fs, n_channels=len(self.args.ch_names))
        ds_val   = LMDBChannelEpochDataset(self.args.base_path, 'val', fs=self.model.backbone.fs, n_channels=len(self.args.ch_names))
        ds_test  = LMDBChannelEpochDataset(self.args.base_path, 'test', fs=self.model.backbone.fs, n_channels=len(self.args.ch_names))

        # Combine ONLY the training and validation sets for pre-training and cross-validation.
        X_train_val = np.concatenate([ds_train.total_x, ds_val.total_x], axis=0)
        Y_train_val_cpc = np.concatenate([ds_train.total_y, ds_val.total_y], axis=0)
        
        train_val_snips = ArraySnippetDataset(X_train_val, Y_train_val_cpc)
        
        print(f"Loaded {len(ds_train)} train, {len(ds_val)} val, and {len(ds_test)} test snippets.")
        print(f"Combined {len(train_val_snips)} snippets for pre-training and CV.")
      
  
        # --- STEP 2: PRE-TRAIN THE NEURONET BACKBONE ---
        # This section performs self-supervised pre-training on the NeuroNet model.
        # It learns general-purpose features from the EEG data without using labels.
        # NOTE: This assumes `self.neuronet` was created in `__init__`.
    
        # Re-instantiate a fresh NeuroNet model for pre-training
        model_kwargs = {
          'fs': self.args.sfreq, 'second': self.args.second, 'time_window': self.args.time_window,
          'time_step': self.args.time_step, 'encoder_embed_dim': self.args.encoder_embed_dim,
          'encoder_heads': self.args.encoder_heads, 'encoder_depths': self.args.encoder_depths,
          'decoder_embed_dim': self.args.decoder_embed_dim, 'decoder_heads': self.args.decoder_heads,
          'decoder_depths': self.args.decoder_depths, 'projection_hidden': self.args.projection_hidden,
          'temperature': self.args.temperature
        }
        self.neuronet = NeuroNet(**model_kwargs).to(device)
    
        print(f"\n=== Starting NeuroNet Pre-training for {self.args.train_epochs} epochs ===")
        pretrain_loader = DataLoader(train_val_snips, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True)
        pretrain_optim = opt.AdamW(self.neuronet.parameters(), lr=self.args.train_base_learning_rate)

        self.neuronet.train()
        for ep in range(self.args.train_epochs):
            total_loss = 0
            pbar = tqdm(pretrain_loader, desc=f"Pre-train Ep {ep+1}/{self.args.train_epochs}", leave=False)
            for x, _ in pbar:
                x = x.to(device)
                pretrain_optim.zero_grad()
            
                # NeuroNet's forward pass for self-supervised learning
                recon_loss, contrastive_loss, _ = self.neuronet(x, mask_ratio=self.args.mask_ratio)
                loss = recon_loss + self.args.alpha * contrastive_loss
            
                loss.backward()
                pretrain_optim.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            avg_loss = total_loss / len(pretrain_loader)
            print(f"Pre-train Epoch {ep+1}/{self.args.train_epochs}, Average Loss: {avg_loss:.4f}")
    
        print("=== Pre-training complete. Building TCM with pre-trained backbone. ===")

        # --- STEP 3: BUILD THE FINAL MODEL FOR THE DOWNSTREAM TASK ---
        # The pre-trained NeuroNet is now used as a frozen feature extractor.
        backbone = NeuroNetEncoderWrapper(
            fs=self.neuronet.fs, second=self.neuronet.second, time_window=self.neuronet.time_window,
            time_step=self.neuronet.time_step, frame_backbone=self.neuronet.frame_backbone,
            patch_embed=self.neuronet.autoencoder.patch_embed, encoder_block=self.neuronet.autoencoder.encoder_block,
            encoder_norm=self.neuronet.autoencoder.encoder_norm, cls_token=self.neuronet.autoencoder.cls_token,
            pos_embed=self.neuronet.autoencoder.pos_embed, final_length=self.neuronet.autoencoder.embed_dim
        )
        tcm_cls = {
            'lstm': LSTM_TCM, 'mha': MHA_TCM,
            'lstm_mha': LSTM_MHA_TCM, 'mamba': MAMBA_TCM
        }[self.args.temporal_context_modules]

        # --- STEP 4: 5-FOLD CROSS-VALIDATION FOR THE TCM HEAD ---
        # We train and validate the TCM head (e.g., Mamba) to find the best model.      
        train_val_seq = LMDBSequenceDataset(train_val_snips, self.args.temporal_context_length, self.args.window_size)       
        all_seq_labels = np.array([train_val_seq[i][1] for i in range(len(train_val_seq))], dtype=np.int64)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        fold_metrics_binary, fold_metrics_cpc = [], []
        
        best_overall_f1 = 0
        best_overall_state = None
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_seq), 1):
            print(f"\n" + "="*40)
            print(f"=== Starting Fold {fold}/5 ===")
            
            # Re-instantiate the model to ensure each fold starts fresh
            self.tcm = tcm_cls(backbone, self.neuronet.autoencoder.embed_dim, self.args.embed_dim).to(device)

            train_sub, val_sub = Subset(train_val_seq, train_idx), Subset(train_val_seq, val_idx)
        
      
            # Set up weighted sampler to handle class imbalance within the fold's training data
            train_labels_for_fold = all_seq_labels[train_idx]
            class_counts_fold = np.bincount(train_labels_for_fold, minlength=self.n_classes)
            weights_per_class = 1.0 / (class_counts_fold + 1e-6)
            sample_weights_fold = weights_per_class[train_labels_for_fold]
            sampler = WeightedRandomSampler(weights=sample_weights_fold, num_samples=len(sample_weights_fold), replacement=True)
            
            tr = DataLoader(train_sub, batch_size=self.args.batch_size, sampler=sampler, drop_last=True)
            vl = DataLoader(val_sub,   batch_size=self.args.batch_size, shuffle=False, drop_last=False)
            self.model.apply(lambda m: isinstance(m, nn.Linear) and hasattr(m, 'reset_parameters') and m.reset_parameters())
            
            # The optimizer will only update the unfrozen parameters of the TCM head
            optim = opt.AdamW(filter(lambda p: p.requires_grad, self.tcm.parameters()), lr=self.args.lr)
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
              
                # Evaluate and print metrics for the current epoch
                print(f"\n--- Fold {fold} Epoch {ep+1} Validation Metrics ---")
                
                # Evaluate both binary and CPC tasks
                metrics_binary = self.evaluate_loader_binary(vl)
                metrics_cpc = self.evaluate_loader_cpc(vl)
                
                # Print binary metrics for the epoch
                print("  Binary Outcome:")
                for k, v in metrics_binary.items():
                    if k != 'confusion_matrix':
                        print(f"    {k}: {v:.4f}")

                # Print CPC metrics for the epoch
                print("  CPC (5-Class) Outcome:")
                for k, v in metrics_cpc.items():
                    if k != 'confusion_matrix':
                        print(f"    {k}: {v:.4f}")
                
                # Use CPC macro F1-score for model selection
                if metrics_cpc['f1'] > best_fold_f1:
                    best_fold_f1 = metrics_cpc['f1']
                    best_fold_state = self.tcm.state_dict()
                    print("    (New best model for this fold saved)")
                print("-" * 40) # Separator for readability
            
            self.tcm.load_state_dict(best_fold_state)
            val_metrics_binary = self.evaluate_loader_binary(vl)
            val_metrics_cpc = self.evaluate_loader_cpc(vl)
            fold_metrics_binary.append(val_metrics_binary)
            fold_metrics_cpc.append(val_metrics_cpc)
            
            # --- Nicer printing for fold results ---
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
        
        # --- Print Mean CV Metrics ---
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

        # --- STEP 5: FINAL EVALUATION ON HELD-OUT TEST SET ---
        # After all folds, load the single best model and evaluate it on the untouched test set.
      
        # --- Final Evaluation on Test Set ---
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
