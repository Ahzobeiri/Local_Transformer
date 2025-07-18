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
    parser.add_argument('--ch_names', default=['Fp1','F7','T3','T5','O1',
                                               'Fp2','F8','T4','T6','O2',
                                               'F3','C3','P3','F4','C4',
                                               'P4','Fz','Cz','Pz'], help='List of EEG channel labels to load (must match your LMDB samples)')
    parser.add_argument('--temporal_context_length', default=20)
    parser.add_argument('--window_size', default=10)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)

    parser.add_argument('--embed_dim', default=256)
    parser.add_argument('--temporal_context_modules', choices=['lstm', 'mha', 'lstm_mha', 'mamba'], default='mamba')
    return parser.parse_args()

class LMDBSequenceDataset(Dataset):
    """
    Wraps per‐channel snippets from LMDBChannelEpochDataset into
    overlapping (or non‐overlapping) sequences of length seq_len.
    Returns:
      x_seq: FloatTensor of shape (seq_len, snippet_dim)
      y     : scalar label (we assume all snippets in the sequence share the same CPC)
    """
    def __init__(self,
                 snippet_ds: Dataset,
                 seq_len: int,
                 step: int = None):
        """
        snippet_ds: instance of LMDBChannelEpochDataset
        seq_len   : number of consecutive snippets per sample
        step      : how far to slide the window; if None, uses non‐overlap=seq_len
        """
        self.snippet_ds = snippet_ds
        self.seq_len    = seq_len
        self.step       = step or seq_len
        # precompute how many sequences we can extract
        total_snips = len(self.snippet_ds)
        # floor so we don’t run off the end
        self.indices = list(range(0, total_snips - seq_len + 1, self.step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        # gather seq_len snippets
        xs, ys = [], []
        for offset in range(self.seq_len):
            x, y = self.snippet_ds[start + offset]
            xs.append(x)
            ys.append(y)
        # stack into (seq_len, snippet_dim)
        x_seq = torch.stack(xs, dim=0)
        # all y’s should be identical—just take the first
        return x_seq, ys[0]

class TemporalContextModule(nn.Module):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__()
        self.backbone = self.freeze_backbone(backbone)
        self.backbone_final_length = backbone_final_length
        self.embed_dim = embed_dim
        self.embed_layer = nn.Sequential(
            nn.Linear(backbone_final_length, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def apply_backbone(self, x):
        out = []
        for x_ in torch.split(x, dim=1, split_size_or_sections=1):
            o = self.backbone(x_.squeeze())
            o = self.embed_layer(o)
            out.append(o)
        out = torch.stack(out, dim=1)
        return out

    @staticmethod
    def freeze_backbone(backbone: nn.Module):
        for name, module in backbone.named_modules():
            if name in ['encoder_block.3.ls1', 'encoder_block.3.drop_path1', 'encoder_block.3.norm2',
                        'encoder_block.3.mlp', 'encoder_block.3.mlp.fc1', 'encoder_block.3.mlp.act',
                        'encoder_block.3.mlp.drop1', 'encoder_block.3.mlp.norm', 'encoder_block.3.mlp.fc2',
                        'encoder_block.3.mlp.drop2', 'encoder_block.3.ls2', 'encoder_block.3.drop_path2',
                        'encoder_norm']:
                for param in module.parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False
        return backbone


class LSTM_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone=backbone, backbone_final_length=backbone_final_length, embed_dim=embed_dim)
        self.rnn_layer = 2
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, num_layers=self.rnn_layer)
        self.fc = nn.Linear(self.embed_dim, 5)

    def forward(self, x):
        x = self.apply_backbone(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class MHA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone=backbone, backbone_final_length=backbone_final_length, embed_dim=embed_dim)
        self.mha_heads = 8
        self.mha_layer = 2
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.embed_dim, self.mha_heads),
                                                 num_layers=self.mha_layer)
        self.fc = nn.Linear(self.embed_dim, 5)

    def forward(self, x):
        x = self.apply_backbone(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x


class LSTM_MHA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone=backbone, backbone_final_length=backbone_final_length, embed_dim=embed_dim)
        self.mha_heads = 8
        self.mha_layer = 2
        self.rnn_layer = 1
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, num_layers=self.rnn_layer)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.embed_dim, self.mha_heads),
                                                 num_layers=self.mha_layer)
        self.fc = nn.Linear(self.embed_dim, 5)

    def forward(self, x):
        x = self.apply_backbone(x)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x


class MAMBA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone=backbone, backbone_final_length=backbone_final_length, embed_dim=embed_dim)
        self.mamba_heads = 8
        self.mamba_layer = 1
        self.mamba = nn.Sequential(*[
            Mamba(d_model=self.embed_dim,
                  d_state=16,
                  d_conv=4,
                  expand=2)
            for _ in range(self.mamba_layer)
        ])
        self.fc = nn.Linear(self.embed_dim, 5)

    def forward(self, x):
        x = self.apply_backbone(x)
        x = self.mamba(x)
        x = self.fc(x)
        return x

class Trainer:
    def __init__(self, args):
        self.args    = args
        ckpt         = torch.load(os.path.join(args.ckpt_path,'model/best_model.pth'), map_location='cpu')
        param        = ckpt['model_parameter']
        pretrained   = NeuroNet(**param)
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
        train_ds = LMDBChannelEpochDataset(self.args.base_path,'train',fs=self.model.backbone.fs,
                                           n_channels=len(self.args.ch_names))
        val_ds   = LMDBChannelEpochDataset(self.args.base_path,'val',fs=self.model.backbone.fs,
                                           n_channels=len(self.args.ch_names))
        test_ds  = LMDBChannelEpochDataset(self.args.base_path,'test',fs=self.model.backbone.fs,
                                           n_channels=len(self.args.ch_names))
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
