# Usage:
#   python tokenizer_bytelevel.py --corpus data/war_and_peace.txt data/Siddhartha.txt data/Shakespeare.txt \
#                                 --vocab_size 4096 --out tokenizer/bpe_tokenizer.json
import argparse
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
# （可选）正则归一化：GPT-2 的 byte-level 通常不做 Unicode 归一化，这里就不加 NFKC 了

def train_bpe_bytelevel(corpus_files, vocab_size=4096, out="bpe_tokenizer.json"):
    # 1) 模型 + 预分词 + 解码器（GPT-2 风格）
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tok.decoder = ByteLevelDecoder()

    # 2) Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,                  # 小于2的pair不学，收敛更稳
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )

    # 3) 训练
    files = [str(Path(f)) for f in corpus_files]
    tok.train(files=files, trainer=trainer)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(out)

    # 4) 打印关键信息
    print(f"✅ saved tokenizer to {out}")
    print("vocab size:", tok.get_vocab_size())
    print("IDs -> PAD:", tok.token_to_id("[PAD]"),
          "UNK:", tok.token_to_id("[UNK]"),
          "BOS:", tok.token_to_id("[BOS]"),
          "EOS:", tok.token_to_id("[EOS]"))

    # 5) 快速回环测试（避免 E if fe l 这类问题）
    sample = "The Eiffel Tower is in Paris. The mystery of life is..."
    ids = tok.encode(sample).ids
    back = tok.decode(ids)
    print("roundtrip ok?:", sample == back)
    if sample != back:
        print("decoded:", back)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, nargs="+",
                    default=["data/corpus_all.txt"],
                    help="one or more text files")
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--out", type=str, default="tokenizer/bpe_tokenizer.json")
    args = ap.parse_args()
    train_bpe_bytelevel(args.corpus, args.vocab_size, args.out)
