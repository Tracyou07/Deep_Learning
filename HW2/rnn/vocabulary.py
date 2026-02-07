import torch

def text_to_onehot(seq, char_to_idx):
    vocab_size = len(char_to_idx)
    onehots = torch.zeros(len(seq), vocab_size)
    for t, ch in enumerate(seq):
        if ch in char_to_idx:
            onehots[t, char_to_idx[ch]] = 1.0
        else:
            pass
    return onehots

with open("Shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocab = sorted(list(set(text)))        
m = len(vocab)                         
print(f"Vocabulary size: {m}")

char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}



# test
sample_seq = text[:32]
X = text_to_onehot(sample_seq, char_to_idx)

print("Input sequence:")
print(repr(sample_seq))
print("\nOne-hot tensor shape:", X.shape)
print( X.sum(dim=1)[:5])  
