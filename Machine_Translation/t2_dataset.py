"""
t2_dataset.py
Loads Multi30K (EN->DE) directly from GitHub (no torchtext required).
Provides vocab, DataLoaders, GloVe and one-hot embedding matrices.
"""

import os
import re
import zipfile
import urllib.request
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ── constants ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
GLOVE_DIM  = 100
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")

MULTI30K_BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
MULTI30K_FILES = {
    "train_src": "train.en.gz",
    "train_tgt": "train.de.gz",
    "valid_src": "val.en.gz",
    "valid_tgt": "val.de.gz",
    "test_src":  "test_2016_flickr.en.gz",
    "test_tgt":  "test_2016_flickr.de.gz",
}

GLOVE_URL  = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP  = os.path.join(DATA_DIR, "glove.6B.zip")

SPECIAL = ["<unk>", "<pad>", "<bos>", "<eos>"]
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
MIN_FREQ = 2


# ── download helpers ──────────────────────────────────────────────────────────

def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _download_multi30k():
    import gzip
    _ensure_dir()
    for key, fname in MULTI30K_FILES.items():
        gz_path  = os.path.join(DATA_DIR, fname)
        txt_path = gz_path.replace(".gz", "")
        if not os.path.exists(txt_path):
            url = MULTI30K_BASE + fname
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, gz_path)
            with gzip.open(gz_path, "rb") as gz_in, open(txt_path, "wb") as f_out:
                f_out.write(gz_in.read())
            os.remove(gz_path)
    print("[Dataset] Multi30K files ready.")


def _download_glove():
    _ensure_dir()
    glove_file = os.path.join(DATA_DIR, f"glove.6B.{GLOVE_DIM}d.txt")
    if os.path.exists(glove_file):
        return
    if not os.path.exists(GLOVE_ZIP):
        print("Downloading GloVe 6B (~860 MB, please wait) ...")
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP)
    print("Extracting GloVe ...")
    with zipfile.ZipFile(GLOVE_ZIP, "r") as z:
        z.extractall(DATA_DIR)
    print("[GloVe] Extraction complete.")


# ── tokeniser ─────────────────────────────────────────────────────────────────

def tokenize(text: str):
    return re.findall(r"[a-zA-ZäöüÄÖÜß0-9']+", text.lower().strip())


# ── vocabulary ────────────────────────────────────────────────────────────────

class Vocab:
    def __init__(self, stoi):
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


def _build_vocab_from_file(path: str, min_freq: int = MIN_FREQ) -> Vocab:
    counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            counter.update(tokenize(line))
    stoi = {s: i for i, s in enumerate(SPECIAL)}
    idx = len(SPECIAL)
    for token, freq in counter.most_common():
        if freq >= min_freq:
            stoi[token] = idx
            idx += 1
    return Vocab(stoi)


def build_vocabs():
    _download_multi30k()
    src_path = os.path.join(DATA_DIR, "train.en")
    tgt_path = os.path.join(DATA_DIR, "train.de")
    src_vocab = _build_vocab_from_file(src_path)
    tgt_vocab = _build_vocab_from_file(tgt_path)
    print(f"[Vocab] EN: {len(src_vocab)}  DE: {len(tgt_vocab)}")
    return src_vocab, tgt_vocab


# ── dataset ───────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, split: str, src_vocab: Vocab, tgt_vocab: Vocab):
        suffix_map = {"train": ("train.en", "train.de"),
                      "valid": ("val.en",   "val.de"),
                      "test":  ("test_2016_flickr.en", "test_2016_flickr.de")}
        src_file, tgt_file = suffix_map[split]
        self.pairs = []
        src_lines = open(os.path.join(DATA_DIR, src_file), encoding="utf-8").readlines()
        tgt_lines = open(os.path.join(DATA_DIR, tgt_file), encoding="utf-8").readlines()
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_ids = [BOS_IDX] + [src_vocab[t] for t in tokenize(src_line)] + [EOS_IDX]
            tgt_ids = [BOS_IDX] + [tgt_vocab[t] for t in tokenize(tgt_line)] + [EOS_IDX]
            self.pairs.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def get_dataloaders(src_vocab, tgt_vocab, batch_size=BATCH_SIZE):
    train_ds = TranslationDataset("train", src_vocab, tgt_vocab)
    val_ds   = TranslationDataset("valid", src_vocab, tgt_vocab)
    test_ds  = TranslationDataset("test",  src_vocab, tgt_vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


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
            glove_stoi[parts[0]] = len(glove_vecs)
            glove_vecs.append(list(map(float, parts[1:])))
    glove_full = torch.tensor(glove_vecs, dtype=torch.float)

    vocab_size = len(vocab)
    matrix = torch.zeros(vocab_size, glove_dim)
    found  = 0
    for token, idx in vocab.get_stoi().items():
        if token in glove_stoi:
            matrix[idx] = glove_full[glove_stoi[token]]
            found += 1
        else:
            matrix[idx] = torch.randn(glove_dim) * 0.01
    print(f"[GloVe] {found}/{vocab_size} tokens found.")
    return matrix


def build_onehot_matrix(vocab_size: int) -> torch.Tensor:
    return torch.eye(vocab_size)
