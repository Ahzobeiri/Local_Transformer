import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.ADformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ADformerLayer
from layers.Embed import TokenChannelEmbedding
import numpy as np


class Model(nn.Module):
    """
    Model class
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in

        ##### CHANGE #######
        self.d_model = configs.d_model
        fft_out_dim = configs.seq_len // 2 + 1
        self.spectral_proj = nn.Linear(fft_out_dim, configs.seq_len)
        ##### CHANGE #######

        # Embedding
        if configs.no_temporal_block and configs.no_channel_block:
            raise ValueError("At least one of the two blocks should be True")
        if configs.no_temporal_block:
            patch_len_list = []
        else:
            patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        if configs.no_channel_block:
            up_dim_list = []
        else:
            up_dim_list = list(map(int, configs.up_dim_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")
        # augmentations are necessary for contrastive pretraining
        if augmentations == ["none"] and "pretrain" in self.task_name:
            augmentations = ["flip", "frequency", "jitter", "mask", "channel", "drop"]

        self.enc_embedding = TokenChannelEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            patch_len_list,
            up_dim_list,
            stride_list,
            configs.dropout,
            augmentations,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ADformerLayer(
                        len(patch_len_list),
                        len(up_dim_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        if self.task_name == "supervised" or self.task_name == "finetune":
            # use for final classification
            self.classifier = nn.Linear(
                configs.d_model * len(patch_num_list) +
                configs.d_model * len(up_dim_list),
                configs.num_class,
            )
        elif self.task_name == "pretrain_lead" or self.task_name == "pretrain_moco":
            # use for LEAD or MOCO pretraining framework
            input_dim = configs.d_model * len(patch_num_list) + configs.d_model * len(up_dim_list)
            self.projection_head = nn.Sequential(
                nn.Linear(
                    input_dim,
                    input_dim * 2
                ),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(input_dim * 2, configs.d_model)
            )

            if self.task_name == "pretrain_moco":
                K = configs.K  # queue size
                feat_dim = configs.d_model  # assuming projection_head output dimension is configs.d_model
                self.register_buffer("queue", torch.randn(feat_dim, K))
                self.queue = F.normalize(self.queue, dim=0)
                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def supervised(self, x_enc, x_mark_enc):

        ##### CHANGE #######
        x_enc_t = x_enc.transpose(1, 2)
        spectral = torch.fft.rfft(x_enc_t, dim=-1, norm='forward')
        spectral = torch.abs(spectral)
        spectral_emb = self.spectral_proj(spectral)
        # print('-----------', spectral_emb.shape)
        # print(x_enc.shape)
        x_enc = x_enc + spectral_emb.transpose(1, 2)

        ##### CHANGE #######
        # Embedding
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        # print(spectral_emb.shape)
        # print(enc_out_t[0].shape)

        ##### CHANGE #######
        # if enc_out_c is not None:
        #     enc_out_c = enc_out_c + spectral_emb
        ##### CHANGE #######

        enc_out_t, enc_out_c, attns_t, attns_c = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        if enc_out_t is None:
            enc_out = enc_out_c
        elif enc_out_c is None:
            enc_out = enc_out_t
        else:
            enc_out = torch.cat((enc_out_t, enc_out_c), dim=1)
        
        # print(enc_out.shape)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.classifier(output)  # (batch_size, num_classes)
        return output

    def pretrain(self, x_enc, x_mark_enc):  # x_enc (batch_size, seq_length, enc_in)
        # Embedding
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        enc_out_t, enc_out_c, attns_t, attns_c = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        if enc_out_t is None:
            enc_out = enc_out_c
        elif enc_out_c is None:
            enc_out = enc_out_t
        else:
            enc_out = torch.cat((enc_out_t, enc_out_c), dim=1)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)

        repr_out = self.projection_head(output)  # (batch_size, repr_len)
        return output, repr_out  # first for downstream tasks, second for encoding representation

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "supervised" or self.task_name == "finetune":
            output = self.supervised(x_enc, x_mark_enc)
            return output
        elif self.task_name == "pretrain_lead" or self.task_name == "pretrain_moco":
            repr_h, repr_z = self.pretrain(x_enc, x_mark_enc)
            return repr_h, repr_z
        else:
            raise ValueError("Task name not recognized or not implemented within the LEAD model")
