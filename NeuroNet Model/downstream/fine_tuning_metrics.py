# -*- coding:utf-8 -*-
import os
import torch
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as opt
from mamba_ssm import Mamba
from models.utils import model_size
from torch.utils.data import Dataset, DataLoader
from models.neuronet.model import NeuroNet, NeuroNetEncoderWrapper
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc, confusion_matrix)
from pretrained.LMDB_data_loader import LMDBChannelEpochDataset
from tqdm import tqdm

warnings.filterwarnings(action='ignore')

# reproducibility
random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default='/projects/scratch/fhajati/ckpt/unimodal/eeg', type=str)
    parser.add_argument('--base_path', default='/projects/scratch/fhajati/physionet.org/files/LMDB_DATA/19Ch_Last1h', type=str)
    parser.add_argument('--ch_names', nargs='+', default=[...])
    parser.add_argument('--temporal_context_length', default=20, type=int)
    parser.add_argument('--window_size', default=10, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--temporal_context_modules', choices=['lstm','mha','lstm_mha','mamba'], default='mamba')
    return parser.parse_args()

class LMDBSequenceDataset(Dataset):
    # same as before...
    ...

class TemporalContextModule(nn.Module):
    # same as before...
    ...

# TCM subclasses: LSTM_TCM, MHA_TCM, etc.
# ...

class Trainer:
    def __init__(self, args):
        self.args    = args
        ckpt         = torch.load(os.path.join(args.ckpt_path,'model/best_model.pth'),map_location='cpu')
        param        = ckpt['model_parameter']
        pretrained   = NeuroNet(**param)
        pretrained.load_state_dict(ckpt['model_state'])
        backbone     = NeuroNetEncoderWrapper(...)
        tcm_cls      = {...}[args.temporal_context_modules]
        self.model   = tcm_cls(backbone,param['embed_dim'],args.embed_dim).to(device)
        self.criterion = nn.CrossEntropyLoss()

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
        # sensitivity = TP/(TP+FN), specificity = TN/(TN+FP)
        TP = cm[1,1]; FN = cm[1,0]
        TN = cm[0,0]; FP = cm[0,1]
        sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
        spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
        return {'accuracy':acc,'f1':f1,'auc':auc_,'aupr':aupr,
                'sensitivity':sens,'specificity':spec,'confusion_matrix':cm}

    def evaluate(self, loader):
        self.model.eval()
        all_true, all_pred, all_prob = [],[],[]
        with torch.no_grad():
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                out = self.model(x)
                probs = nn.functional.softmax(out[:,-1,:], dim=-1)
                pred  = probs.argmax(dim=-1)
                # for multiclass, take average probability of true class
                prob_true = probs.cpu().numpy()[np.arange(len(y)), y.cpu().numpy()]
                all_true.extend(y.cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                all_prob.extend(prob_true)
        return self.compute_metrics(all_true, all_prob, all_pred)

    def train(self):
        # load and wrap datasets
        train_ds = LMDBChannelEpochDataset(self.args.base_path,'train',fs=...,n_channels=...)
        val_ds   = LMDBChannelEpochDataset(self.args.base_path,'val',fs=...,n_channels=...)
        test_ds  = LMDBChannelEpochDataset(self.args.base_path,'test',fs=...,n_channels=...)
        train_seq= LMDBSequenceDataset(train_ds,self.args.temporal_context_length,self.args.window_size)
        val_seq  = LMDBSequenceDataset(val_ds,  self.args.temporal_context_length,self.args.window_size)
        test_seq = LMDBSequenceDataset(test_ds, self.args.temporal_context_length,self.args.window_size)

        train_loader = DataLoader(train_seq,batch_size=self.args.batch_size,shuffle=True,drop_last=True)
        val_loader   = DataLoader(val_seq,  batch_size=self.args.batch_size,shuffle=False,drop_last=False)
        test_loader  = DataLoader(test_seq, batch_size=self.args.batch_size,shuffle=False,drop_last=False)

        optimizer = opt.AdamW(self.model.parameters(), lr=self.args.lr)
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs)

        best_val = 0.0
        for epoch in range(self.args.epochs):
            self.model.train()
            for x,y in tqdm(train_loader,desc=f"Train E{epoch+1}"):
                optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out[:,-1,:], y.to(device))
                loss.backward(); optimizer.step()
            scheduler.step()

            val_metrics = self.evaluate(val_loader)
            print(f"Epoch {epoch+1} Validation:")
            print(val_metrics)
            if val_metrics['f1']>best_val:
                best_val=val_metrics['f1']
                best_state=self.model.state_dict()

        self.model.load_state_dict(best_state)
        print("Test set performance:")
        test_metrics=self.evaluate(test_loader)
        print(test_metrics)

if __name__=='__main__':
    args=get_args()
    Trainer(args).train()
