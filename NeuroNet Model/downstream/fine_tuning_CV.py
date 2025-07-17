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
from sklearn.metrics import accuracy_score, f1_score
from pretrained.LMDB_data_loader import LMDBChannelEpochDataset
from tqdm import tqdm
from sklearn.model_selection import KFold




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
    parser.add_argument('--ch_names', default=['Fp1','F7','T3','T5','O1',
                                               'Fp2','F8','T4','T6','O2',
                                               'F3','C3','P3','F4','C4',
                                               'P4','Fz','Cz','Pz'], help='List of EEG channel labels to load (must match your LMDB samples)')
    parser.add_argument('--temporal_context_length', default=20, type=int)
    parser.add_argument('--window_size', default=10, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--temporal_context_modules', choices=['lstm', 'mha', 'lstm_mha', 'mamba'], default='mamba')
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
    Wraps per-channel snippets from LMDBChannelEpochDataset into overlapping sequences of length seq_len.
    Returns x_seq: (seq_len, snippet_dim), y: scalar label (we assume all snippets in the sequence share the same CPC)
    """
    def __init__(self, snippet_ds: Dataset, seq_len: int, step: int = None):
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


class Trainer(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ckpt_path = os.path.join(self.args.ckpt_path, 'model', 'best_model.pth')
        self.ckpt = torch.load(self.ckpt_path, map_location='cpu')
        self.sfreq = self.ckpt['hyperparameter']['sfreq']
        self.model = self.get_pretrained_model().to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        print('Checkpoint File Path : {}'.format(self.ckpt_path))

        '''
        # 1) base snippet dataset
        snippet_train_ds = LMDBChannelEpochDataset(
            lmdb_path=self.args.base_path,
            mode='train',
            fs=self.sfreq,
            n_channels=len(self.args.ch_names),
        )
        
        snippet_val_ds   = LMDBChannelEpochDataset(
            lmdb_path=self.args.base_path,
            mode='val',
            fs=self.sfreq,
            n_channels=len(self.args.ch_names),
        )
        
         # 2) wrap into sequences of length `temporal_context_length`
        train_dataset = LMDBSequenceDataset(
            snippet_ds=snippet_train_ds,
            seq_len=self.args.temporal_context_length,
            step=self.args.window_size
        )
        eval_dataset  = LMDBSequenceDataset(
            snippet_ds= snippet_val_ds,
            seq_len=self.args.temporal_context_length,
            step=self.args.window_size
        )
                
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      drop_last=True)
        eval_dataloader  = DataLoader(eval_dataset,
                                      batch_size=self.args.batch_size,
                                      shuffle=False,
                                      drop_last=True)

        # Maybe need "test loader".
        '''
        
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

        # Until Here .......................
        #####################################
        
        best_model_state, best_mf1 = None, 0.0
        best_pred, best_real = [], []

        for epoch in range(self.args.epochs):
            print(f"Starting epoch {epoch+1}/{self.args.epochs}")

            # -- Training --
            self.model.train()
            epoch_train_loss = []
            # for batch in train_dataloader:
            for (x, y) in tqdm(train_dataloader, desc=" Training"):
                self.optimizer.zero_grad()
                # x, y = batch
                x, y = x.to(device), y.to(device)

                out = self.model(x)
                loss, pred, real = self.get_loss(out, y)

                epoch_train_loss.append(float(loss.detach().cpu().item()))
                loss.backward()
                self.optimizer.step()


            # -- Validation --
            self.model.eval()
            epoch_test_loss = []
            epoch_real, epoch_pred = [], []
            for x, y in tqdm(eval_dataloader, desc="  Validation", leave=False):
                # x, y = batch
                x, y = x.to(device), y.to(device)
                try:
                    out = self.model(x)
                except IndexError:
                    continue
                loss, pred, real = self.get_loss(out, y)
                pred = torch.argmax(pred, dim=-1)
                epoch_real.extend(list(real.detach().cpu().numpy()))
                epoch_pred.extend(list(pred.detach().cpu().numpy()))
                epoch_test_loss.append(float(loss.detach().cpu().item()))

            epoch_train_loss, epoch_test_loss = np.mean(epoch_train_loss), np.mean(epoch_test_loss)
            eval_acc, eval_mf1 = accuracy_score(y_true=epoch_real, y_pred=epoch_pred), \
                                 f1_score(y_true=epoch_real, y_pred=epoch_pred, average='macro')

            print('[Epoch] : {0:03d} \t '
                  '[Train Loss] => {1:.4f} \t '
                  '[Evaluation Loss] => {2:.4f} \t '
                  '[Evaluation Accuracy] => {3:.4f} \t'
                  '[Evaluation Macro-F1] => {4:.4f}'.format(epoch + 1, epoch_train_loss, epoch_test_loss,
                                                            eval_acc, eval_mf1))

            if best_mf1 < eval_mf1:
                best_mf1 = eval_mf1
                best_model_state = self.model.state_dict()
                best_pred, best_real = epoch_pred, epoch_real

            self.scheduler.step()

        self.save_ckpt(best_model_state, best_pred, best_real)

    def save_ckpt(self, model_state, pred, real):
        if not os.path.exists(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'fine_tuning')):
            os.makedirs(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'fine_tuning'))

        save_path = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'fine_tuning', 'best_model.pth')
        torch.save({
            'backbone_name': 'NeuroNet_FineTuning',
            'model_state': model_state,
            'hyperparameter': self.args.__dict__,
            'result': {'real': real, 'pred': pred},
            'paths': {'train_paths': self.ft_paths, 'eval_paths': self.eval_paths}
        }, save_path)

    def get_pretrained_model(self):
        # 1. Prepared Pretrained Model
        model_parameter = self.ckpt['model_parameter']
        pretrained_model = NeuroNet(**model_parameter)
        pretrained_model.load_state_dict(self.ckpt['model_state'])

        # 2. Encoder Wrapper
        backbone = NeuroNetEncoderWrapper(
            fs=model_parameter['fs'], second=model_parameter['second'],
            time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
            frame_backbone=pretrained_model.frame_backbone,
            patch_embed=pretrained_model.autoencoder.patch_embed,
            encoder_block=pretrained_model.autoencoder.encoder_block,
            encoder_norm=pretrained_model.autoencoder.encoder_norm,
            cls_token=pretrained_model.autoencoder.cls_token,
            pos_embed=pretrained_model.autoencoder.pos_embed,
            final_length=pretrained_model.autoencoder.embed_dim,
        )

        # 3. Temporal Context Module
        tcm = self.get_temporal_context_module()
        model = tcm(backbone=backbone,
                    backbone_final_length=pretrained_model.autoencoder.embed_dim,
                    embed_dim=self.args.embed_dim)
        return model

    def get_temporal_context_module(self):
        if self.args.temporal_context_modules == 'lstm':
            return LSTM_TCM
        if self.args.temporal_context_modules == 'mha':
            return MHA_TCM
        if self.args.temporal_context_modules == 'lstm_mha':
            return LSTM_MHA_TCM
        if self.args.temporal_context_modules == 'mamba':
            return MAMBA_TCM

    def get_loss(self, pred, real):

        '''
        if pred.dim() == 3:
            pred = pred.view(-1, pred.size(2))
            real = real.view(-1)
        '''
        # pred: (batch, seq_len, n_classes), real:(batch, )
        pred_last =pred[:, -1, :]
        loss = self.criterion(pred_last, real)
        return loss, pred_last, real




'''
class TorchDataset(Dataset):
    def __init__(self, paths, temporal_context_length, window_size,
                 sfreq: int, rfreq: int):
        super().__init__()
        self.sfreq, self.rfreq = sfreq, rfreq
        self.info = mne.create_info(sfreq=sfreq, ch_types='eeg', ch_names=['Fp1'])
        self.x, self.y = self.get_data(paths,
                                       temporal_context_length=temporal_context_length,
                                       window_size=window_size)
        self.x, self.y = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long)

    def get_data(self, paths, temporal_context_length, window_size):
        total_x, total_y = [], []
        for path in paths:
            data = np.load(path)
            x, y = data['x'], data['y']
            x = np.expand_dims(x, axis=1)
            x = mne.EpochsArray(x, info=self.info)
            x = x.resample(self.rfreq)
            x = x.get_data().squeeze()
            x = self.many_to_many(x, temporal_context_length, window_size)
            y = self.many_to_many(y, temporal_context_length, window_size)
            total_x.append(x)
            total_y.append(y)
        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)
        return total_x, total_y

    @staticmethod
    def many_to_many(elements, temporal_context_length, window_size):
        size = len(elements)
        total = []
        if size <= temporal_context_length:
            return elements
        for i in range(0, size-temporal_context_length+1, window_size):
            temp = np.array(elements[i:i+temporal_context_length])
            total.append(temp)
        total.append(elements[size-temporal_context_length:size])
        total = np.array(total)
        return total

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = torch.tensor(self.x[item])
        y = torch.tensor(self.y[item])
        return x, y
'''

if __name__ == '__main__':
    augments = get_args()
    for n_fold in range(10):
        augments.n_fold = n_fold
        trainer = Trainer(augments)
        trainer.train()
