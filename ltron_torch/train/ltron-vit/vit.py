""" Models """

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import matplotlib.pyplot as plt


class LTron_ViT(nn.Module):

    def __init__(
        self, 
        lr=1e-4, 
        weight_decay=0.0, 
        embedding_dim=256, 
        patch_size=8*8*3, 
        optimizer_type="Adam", 
        device=None
    ):
        super().__init__()

        self.sequence_encoder = TransformerEncoder(embedding_dim=embedding_dim)
        self.patch_encoder = PatchEncoder(embedding_dim=embedding_dim, patch_size=patch_size)
        self.category_decoder = PredictionDecoder(120, embedding_dim=embedding_dim)

        self.positional_encoding = PositionalEncoding(embedding_dim)
        if optimizer_type == "Adam":
            self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device

    def train_epoch(self, dataloader):
        losses = {"classification": []}
        self.train()
        for batch in dataloader:
            self.optim.zero_grad()

            ### TODO: Add Masking
            img = batch["images"].to(self.device).float() ### TODO: Figure out why I have to set this to .float() here
            img_positions = batch["image_positions"].to(self.device)
            category_label = batch["categories"].to(self.device)

            # Flatten batch and seq into just batch
            flat_img = img.view(img.shape[0]*img.shape[1], *img.shape[2:])
            patch_embedding = self.patch_encoder(flat_img)
            # Fold batch back into batch and seq and swap batch and seq dimensions for transformer
            patch_embedding_sequence = patch_embedding.view(img.shape[0], img.shape[1], -1).permute(1,0,2)
            position_embedding_sequence = img_positions.view(img.shape[0], img.shape[1], -1).permute(1,0,2)
            image_encoding = self.sequence_encoder(self.positional_encoding(patch_embedding_sequence, position_embedding_sequence))
            # Flatten batch and seq into into just batch, and permute dimensions back
            flat_image_encoding = image_encoding.view(img.shape[1] * img.shape[0], -1)

            # Predict brick classification
            category_prediction = self.category_decoder(flat_image_encoding)
            classification_loss = F.cross_entropy(category_prediction, category_label.view(-1))
            losses["classification"].append(classification_loss.item())

            loss = classification_loss
            loss.backward()

            self.optim.step()

        return {k: np.mean(v) for k,v in losses.items()}

    def test_epoch(self, dataloader):
        self.eval()
        with torch.no_grad():
            losses = {"classification": []}
            for batch in dataloader:
                img = batch["images"].to(self.device).float()
                img_positions = batch["image_positions"].to(self.device)
                category_label = batch["categories"].to(self.device)

                # Flatten batch and seq into just batch
                flat_img = img.view(img.shape[0]*img.shape[1], *img.shape[2:])
                patch_embedding = self.patch_encoder(flat_img)
                # Fold batch back into batch and seq and swap batch and seq dimensions for transformer
                patch_embedding_sequence = patch_embedding.view(img.shape[0], img.shape[1], -1).permute(1,0,2)
                position_embedding_sequence = img_positions.view(img.shape[0], img.shape[1], -1).permute(1,0,2)
                image_encoding = self.sequence_encoder(self.positional_encoding(patch_embedding_sequence, position_embedding_sequence))
                # Flatten batch and seq into into just batch, and permute dimensions back
                flat_image_encoding = image_encoding.view(img.shape[1] * img.shape[0], -1)

                # Predict brick classification
                category_prediction = self.category_decoder(flat_image_encoding)
                classification_loss = F.cross_entropy(category_prediction, category_label.view(-1))
                losses["classification"].append(classification_loss.item())

            return {k: np.mean(v) for k,v in losses.items()}


class PatchEncoder(nn.Module):

    def __init__(self, patch_size=8*8, embedding_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class PredictionDecoder(nn.Module):

    def __init__(self, out_size, embedding_dim=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, out_size),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, embedding_dim=512, num_layers=2):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, 8)
        self.model = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        return self.model(x)


class TransformerDecoder(nn.Module):

    def __init__(self, embedding_dim=512, num_layers=2):
        super().__init__()
        decoder_layers = nn.TransformerDecoderLayer(embedding_dim, 8)
        self.model = nn.TransformerDecoder(decoder_layers, num_layers)

    def forward(self, x, a):
        return self.model(x, a)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_img_len=32*32, max_seq_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        img_d = d_model // 2
        img_pe = torch.zeros(max_img_len, img_d)
        position = torch.arange(0, max_img_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, img_d, 2).float() * (-math.log(10000.0) / img_d))
        img_pe[:, 0:    :2] = torch.sin(position * div_term)
        img_pe[:, 1::2] = torch.cos(position * div_term)
        # img_pe = img_pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('img_pe', img_pe)

        seq_d = d_model // 2
        seq_pe = torch.zeros(max_seq_len, seq_d)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, seq_d, 2).float() * (-math.log(10000.0) / seq_d))
        seq_pe[:, 0:    :2] = torch.sin(position * div_term)
        seq_pe[:, 1::2] = torch.cos(position * div_term)
        # seq_pe = seq_pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('seq_pe', img_pe)

    def forward(self, x, pos):
        x = x + torch.cat((self.img_pe[pos[:,:,0]], self.seq_pe[pos[:,:,1]]), dim=2)
        return self.dropout(x)
