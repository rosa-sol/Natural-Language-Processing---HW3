"""
t2_architecture.py
Encoder-Decoder architectures for machine translation:
  - GRUSeq2Seq  : GRU-based encoder and decoder
  - RNNSeq2Seq  : Basic Elman RNN encoder and decoder

Both share the same Encoder / Decoder base structure so they can be
swapped in and out cleanly in the training and evaluation scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from t2_dataset import PAD_IDX


# ── Encoder ───────────────────────────────────────────────────────────────────

class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout,
                 pretrained_emb=None, freeze_emb=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(pretrained_emb, requires_grad=not freeze_emb)

        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        src : (T_src, B) LongTensor
        Returns
        -------
        outputs : (T_src, B, H)
        hidden  : (num_layers, B, H)
        """
        emb = self.dropout(self.embedding(src))       # (T, B, E)
        outputs, hidden = self.gru(emb)
        return outputs, hidden


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout,
                 pretrained_emb=None, freeze_emb=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(pretrained_emb, requires_grad=not freeze_emb)

        self.rnn = nn.RNN(
            embed_dim, hidden_dim, num_layers,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(emb)
        return outputs, hidden


# ── Decoder ───────────────────────────────────────────────────────────────────

class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout,
                 pretrained_emb=None, freeze_emb=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(pretrained_emb, requires_grad=not freeze_emb)

        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.dropout  = nn.Dropout(dropout)
        self.fc_out   = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_token, hidden):
        """
        tgt_token : (1, B) — single time step
        hidden    : (num_layers, B, H)

        Returns
        -------
        logits : (B, vocab_size)
        hidden : (num_layers, B, H)
        """
        emb = self.dropout(self.embedding(tgt_token))  # (1, B, E)
        out, hidden = self.gru(emb, hidden)             # (1, B, H)
        logits = self.fc_out(out.squeeze(0))            # (B, V)
        return logits, hidden


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout,
                 pretrained_emb=None, freeze_emb=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(pretrained_emb, requires_grad=not freeze_emb)

        self.rnn = nn.RNN(
            embed_dim, hidden_dim, num_layers,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out  = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_token, hidden):
        emb = self.dropout(self.embedding(tgt_token))
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc_out(out.squeeze(0))
        return logits, hidden


# ── Seq2Seq wrappers ──────────────────────────────────────────────────────────

class Seq2Seq(nn.Module):
    """
    Generic Seq2Seq wrapper that pairs any Encoder with any Decoder.
    Supports teacher forcing during training.
    """

    def __init__(self, encoder, decoder, tgt_vocab_size, device):
        super().__init__()
        self.encoder       = encoder
        self.decoder       = decoder
        self.tgt_vocab_size = tgt_vocab_size
        self.device        = device

    def forward(self, src, tgt, teacher_forcing_ratio: float = 0.5):
        """
        src : (T_src, B)
        tgt : (T_tgt, B)

        Returns
        -------
        outputs : (T_tgt, B, tgt_vocab_size)
        """
        T_tgt, B = tgt.shape
        outputs = torch.zeros(T_tgt, B, self.tgt_vocab_size).to(self.device)

        _, hidden = self.encoder(src)

        # first decoder input is the <bos> token
        dec_input = tgt[0].unsqueeze(0)   # (1, B)

        for t in range(1, T_tgt):
            logits, hidden = self.decoder(dec_input, hidden)   # (B, V)
            outputs[t] = logits
            # teacher forcing: use ground truth or model prediction
            if torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = tgt[t].unsqueeze(0)
            else:
                dec_input = logits.argmax(dim=-1).unsqueeze(0)

        return outputs


# ── convenience constructors ──────────────────────────────────────────────────

def build_gru_seq2seq(
    src_vocab_size, tgt_vocab_size,
    src_embed_dim, tgt_embed_dim,
    hidden_dim, num_layers, dropout, device,
    src_pretrained=None, tgt_pretrained=None,
):
    encoder = GRUEncoder(src_vocab_size, src_embed_dim, hidden_dim, num_layers, dropout,
                         pretrained_emb=src_pretrained)
    decoder = GRUDecoder(tgt_vocab_size, tgt_embed_dim, hidden_dim, num_layers, dropout,
                         pretrained_emb=tgt_pretrained)
    return Seq2Seq(encoder, decoder, tgt_vocab_size, device).to(device)


def build_rnn_seq2seq(
    src_vocab_size, tgt_vocab_size,
    src_embed_dim, tgt_embed_dim,
    hidden_dim, num_layers, dropout, device,
    src_pretrained=None, tgt_pretrained=None,
):
    encoder = RNNEncoder(src_vocab_size, src_embed_dim, hidden_dim, num_layers, dropout,
                         pretrained_emb=src_pretrained)
    decoder = RNNDecoder(tgt_vocab_size, tgt_embed_dim, hidden_dim, num_layers, dropout,
                         pretrained_emb=tgt_pretrained)
    return Seq2Seq(encoder, decoder, tgt_vocab_size, device).to(device)
