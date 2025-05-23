import numpy as np, string, json, pathlib as p
ALPHABET = ''.join(chr(i) for i in range(32,127))
PAD, UNK = 0, 1
VOCAB = {c:i+2 for i,c in enumerate(ALPHABET)}

def encode(url:str, max_len=200):
    ids = np.full(max_len, PAD, np.uint8)
    for i,ch in enumerate(url[:max_len]):
        ids[i] = VOCAB.get(ch, UNK)
    return ids
