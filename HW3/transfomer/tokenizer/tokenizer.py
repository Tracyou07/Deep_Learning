# Usage: python tokenizer.py --corpus corpus_all.txt --vocab_size 2048 --out bpe_tokenizer.json
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.normalizers import NFKC

def train_bpe(corpus, vocab_size=2048, out="bpe_tokenizer.json"):
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = BertPreTokenizer()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[PAD]"])
    tok.train(files=[corpus], trainer=trainer)
    tok.save(out)
    print(f"saved tokenizer to {out}")
    print("PAD id:", tok.token_to_id("[PAD]"), "CLS id:", tok.token_to_id("[CLS]"), "UNK id:", tok.token_to_id("[UNK]"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="corpus_all.txt")
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--out", type=str, default="bpe_tokenizer.json")
    args = ap.parse_args()
    train_bpe(args.corpus, args.vocab_size, args.out)
