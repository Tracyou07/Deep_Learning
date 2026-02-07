# dataset_sentences.py
# 句子级语言建模数据管道：分句 → 编码 → 动态 padding → DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import re
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

#Make sentence
def simple_sentence_split(text: str) -> List[str]:
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\n+', '. ', text)              
    sents = re.split(r'(?<=[.!?])\s+', text)       
    return [s.strip() for s in sents if s.strip()]

def load_sentences_from_files(file_list: List[Path]) -> List[str]:
    all_sents = []
    for p in file_list:
        if not Path(p).exists():
            raise FileNotFoundError(f"File not found: {p}")
        text = Path(p).read_text(encoding="utf-8", errors="ignore")
        all_sents.extend(simple_sentence_split(text))
    return [s for s in all_sents if len(s) > 1]


# sentence with id
class SentenceDataset(Dataset):
    def __init__(self, sentences: List[str], tokenizer: Tokenizer):
        super().__init__()
        self.tok = tokenizer
        self.sentences = sentences

    def __len__(self): return len(self.sentences)

    def __getitem__(self, idx) -> List[int]:
        return self.tok.encode(self.sentences[idx]).ids

def pad_collate(batch_ids: List[List[int]], pad_id: int, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    dynamic padding
    """
    batch_ids = [ids[:T] for ids in batch_ids if len(ids) > 0]
    if not batch_ids:
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long)

    L = max(len(ids) for ids in batch_ids)
    xs, ys = [], []
    for ids in batch_ids:
        inp = ids
        tgt = ids[1:] + [pad_id]
        if len(inp) < L:
            pad_n = L - len(inp)
            inp = inp + [pad_id] * pad_n
            tgt = tgt + [pad_id] * pad_n
        xs.append(torch.tensor(inp, dtype=torch.long))
        ys.append(torch.tensor(tgt, dtype=torch.long))
    return torch.stack(xs, 0), torch.stack(ys, 0)


# ---------- 总入口：构造 DataLoader ----------

def make_sentence_loaders(
    tokenizer_path: Path,
    train_files: List[Path],
    test_files: Optional[List[Path]] = None,
    T: int = 128,
    batch_size: int = 64,
    workers: int = 2,
    val_ratio: float = 0.05,
    max_train_sents: Optional[int] = None,
    max_val_sents: Optional[int] = 4000,
    max_test_sents: Optional[int] = 8000,
):
    """
    return
      tok, PAD, V, train_loader, val_loader, test_loader, stats(dict)
    """
    tok = Tokenizer.from_file(str(tokenizer_path))
    PAD = tok.token_to_id("[PAD]")
    V = tok.get_vocab_size()

    # files → sentence
    train_sents = load_sentences_from_files(train_files)
    val_split = int(len(train_sents) * val_ratio)
    val_sents = train_sents[-val_split:] if val_split > 0 else []
    train_sents = train_sents[:-val_split] if val_split > 0 else train_sents

    test_sents = load_sentences_from_files(test_files) if test_files else []

    # 可选：裁剪规模（更快训练与评估）
    import random
    random.shuffle(train_sents); random.shuffle(val_sents); random.shuffle(test_sents)
    if max_train_sents is not None: train_sents = train_sents[:max_train_sents]
    if max_val_sents   is not None: val_sents   = val_sents[:max_val_sents]
    if max_test_sents  is not None: test_sents  = test_sents[:max_test_sents]

    train_ds = SentenceDataset(train_sents, tok)
    val_ds   = SentenceDataset(val_sents, tok) if val_sents else None
    test_ds  = SentenceDataset(test_sents, tok) if test_sents else None

    collate_fn = partial(pad_collate, pad_id=PAD, T=T)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, collate_fn=collate_fn, drop_last=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, collate_fn=collate_fn, drop_last=False) if test_ds else None

    stats = dict(n_train=len(train_sents), n_val=len(val_sents), n_test=len(test_sents))
    return tok, PAD, V, train_loader, val_loader, test_loader, stats
