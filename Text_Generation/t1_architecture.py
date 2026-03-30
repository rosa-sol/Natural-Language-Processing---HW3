"""
t1_architecture.py
Defines two language model architectures:
  - GRULanguageModel
  - RNNLanguageModel  (basic Elman RNN)

Both accept either a pre-trained GloVe embedding matrix or a one-hot matrix,
passed in at construction time so the dataset module controls embedding choice.
"""

import torch
import torch.nn as nn


class GRULanguageModel(nn.Module):
    """
    GRU-based language model.

    Parameters
    ----------
    vocab_size      : int   - number of tokens in the vocabulary
    embed_dim       : int   - size of the input embedding
    hidden_dim      : int   - number of GRU hidden units
    num_layers      : int   - number of stacked GRU layers
    dropout         : float - dropout probability (applied between layers)
    pretrained_emb  : Tensor or None
                      If provided, initialises the embedding layer with these
                      weights.  Shape must be (vocab_size, embed_dim).
    freeze_emb      : bool  - whether to freeze pre-trained embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_emb: torch.Tensor = None,
        freeze_emb: bool = False,
    ):
        super().__init__()

        # ── embedding ────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_emb is not None:
            assert pretrained_emb.shape == (vocab_size, embed_dim), (
                f"Embedding matrix shape mismatch: "
                f"expected ({vocab_size}, {embed_dim}), "
                f"got {tuple(pretrained_emb.shape)}"
            )
            self.embedding.weight = nn.Parameter(pretrained_emb, requires_grad=not freeze_emb)

        # ── recurrent core ───────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,   # expects (T, B, E)
        )

        self.dropout = nn.Dropout(dropout)

        # ── output projection ────────────────────────────────────────────────
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None):
        """
        Parameters
        ----------
        x      : (T, B) LongTensor of token indices
        hidden : (num_layers, B, hidden_dim) or None

        Returns
        -------
        logits : (T*B, vocab_size)
        hidden : (num_layers, B, hidden_dim)
        """
        emb = self.dropout(self.embedding(x))          # (T, B, E)
        out, hidden = self.gru(emb, hidden)             # (T, B, H)
        out = self.dropout(out)
        logits = self.fc(out.view(-1, out.size(2)))     # (T*B, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        weight = next(self.parameters())
        return weight.new_zeros(self.gru.num_layers, batch_size, self.gru.hidden_size).to(device)


# ─────────────────────────────────────────────────────────────────────────────

class RNNLanguageModel(nn.Module):
    """
    Basic Elman RNN language model.
    Same interface as GRULanguageModel for drop-in comparison.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_emb: torch.Tensor = None,
        freeze_emb: bool = False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_emb is not None:
            assert pretrained_emb.shape == (vocab_size, embed_dim)
            self.embedding.weight = nn.Parameter(pretrained_emb, requires_grad=not freeze_emb)

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.rnn(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out.view(-1, out.size(2)))
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        weight = next(self.parameters())
        return weight.new_zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device)
