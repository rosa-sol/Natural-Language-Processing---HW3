"""
t1_dataset.py
Loads WikiText-2 directly from the web (no torchtext required).
Provides GloVe and one-hot embedding matrices.
"""

import os
import re
import zipfile
import urllib.request
import torch
from collections import Counter


# ── constants ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
SEQ_LEN    = 35
GLOVE_DIM  = 100
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")

WIKITEXT2_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/{}"
)
SPLITS = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}

GLOVE_URL  = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP  = os.path.join(DATA_DIR, "glove.6B.zip")
GLOVE_FILE = os.path.join(DATA_DIR, f"glove.6B.{GLOVE_DIM}d.txt")

SPECIAL = ["<unk>", "<pad>", "<bos>", "<eos>"]
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


# ── download helpers ──────────────────────────────────────────────────────────

def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _download_wikitext2():
    _ensure_dir()
    for split, fname in SPLITS.items():
        path = os.path.join(DATA_DIR, f"wikitext2_{fname}")
        if not os.path.exists(path):
            url = WIKITEXT2_URL.format(fname)
            print(f"Downloading WikiText-2 {split} split ...")
            urllib.request.urlretrieve(url, path)
    print("[Dataset] WikiText-2 files ready.")


def _download_glove():
    _ensure_dir()
    if os.path.exists(GLOVE_FILE):
        return
    if not os.path.exists(GLOVE_ZIP):
        print("Downloading GloVe 6B (this is ~860 MB, please wait) ...")
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP)
    print("Extracting GloVe ...")
    with zipfile.ZipFile(GLOVE_ZIP, "r") as z:
        z.extractall(DATA_DIR)
    print("[GloVe] Extraction complete.")


# ── tokeniser ─────────────────────────────────────────────────────────────────

def tokenize(text: str):
    text = text.lower().strip()
    return re.findall(r"[a-z0-9']+", text)


# ── vocabulary ────────────────────────────────────────────────────────────────

class Vocab:
    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, UNK_IDX)

    def get_itos(self):
        return [self.itos[i] for i in range(len(self.itos))]

    def get_stoi(self):
        return self.stoi


def build_vocab(min_freq: int = 3) -> Vocab:
    _download_wikitext2()
    counter = Counter()
    for fname in SPLITS.values():
        path = os.path.join(DATA_DIR, f"wikitext2_{fname}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                counter.update(tokenize(line))

    stoi = {s: i for i, s in enumerate(SPECIAL)}
    idx = len(SPECIAL)
    for token, freq in counter.most_common():
        if freq >= min_freq:
            stoi[token] = idx
            idx += 1

    print(f"[Vocab] Size: {len(stoi)}")
    return Vocab(stoi)


# ── encoding ──────────────────────────────────────────────────────────────────

def encode_split(split: str, vocab: Vocab) -> torch.Tensor:
    fname = os.path.join(DATA_DIR, f"wikitext2_{SPLITS[split]}")
    ids = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            tokens = tokenize(line)
            if tokens:
                ids += [vocab[t] for t in tokens]
    return torch.tensor(ids, dtype=torch.long)


def batchify(data: torch.Tensor, bsz: int) -> torch.Tensor:
    n = data.size(0) // bsz
    data = data[: n * bsz]
    return data.view(bsz, -1).t().contiguous()


def get_batch(source: torch.Tensor, i: int, seq_len: int = SEQ_LEN):
    length = min(seq_len, source.size(0) - 1 - i)
    x = source[i : i + length]
    y = source[i + 1 : i + 1 + length]
    return x, y


# ── embedding helpers ─────────────────────────────────────────────────────────

def build_glove_matrix(vocab: Vocab, glove_dim: int = GLOVE_DIM) -> torch.Tensor:
    _download_glove()
    glove_path = os.path.join(DATA_DIR, f"glove.6B.{glove_dim}d.txt")

    glove_stoi = {}
    glove_vecs = []
    print("Loading GloVe vectors ...")
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word  = parts[0]
            vec   = list(map(float, parts[1:]))
            glove_stoi[word] = len(glove_vecs)
            glove_vecs.append(vec)
    glove_matrix_full = torch.tensor(glove_vecs, dtype=torch.float)

    vocab_size = len(vocab)
    matrix = torch.zeros(vocab_size, glove_dim)
    found  = 0
    for token, idx in vocab.get_stoi().items():
        if token in glove_stoi:
            matrix[idx] = glove_matrix_full[glove_stoi[token]]
            found += 1
        else:
            matrix[idx] = torch.randn(glove_dim) * 0.01

    print(f"[GloVe] {found}/{vocab_size} tokens initialised from pre-trained vectors.")
    return matrix


def build_onehot_matrix(vocab_size: int) -> torch.Tensor:
    return torch.eye(vocab_size)


# ── convenience loader ────────────────────────────────────────────────────────

def load_data(batch_size: int = BATCH_SIZE):
    vocab      = build_vocab()
    train_data = batchify(encode_split("train", vocab), batch_size)
    val_data   = batchify(encode_split("valid", vocab), batch_size)
    test_data  = batchify(encode_split("test",  vocab), batch_size)

    print(f"[Dataset] Train: {train_data.shape} | Val: {val_data.shape} | Test: {test_data.shape}")
    return vocab, train_data, val_data, test_data
