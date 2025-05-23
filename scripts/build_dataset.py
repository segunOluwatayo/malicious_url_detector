# scripts/build_dataset.py
"""
Create:
  data/dataset.npz
  data/scaler.json
Run directly – importing this module NO LONGER triggers any work.
"""
import json, math, tldextract, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from split import group_split
from feeds import DATA

RAW = DATA / "raw.csv"

# ────────────── helpers ─────────────────────────────────────────────── #
ALPHABET = ''.join(chr(i) for i in range(32, 127))
PAD, UNK = 0, 1
VOCAB = {c: i + 2 for i, c in enumerate(ALPHABET)}

def encode(url: str, max_len: int = 200) -> np.ndarray:
    dom = tldextract.extract(url).top_domain_under_public_suffix or url
    arr = np.full(max_len, PAD, np.uint8)
    for i, ch in enumerate(dom[:max_len]):
        arr[i] = VOCAB.get(ch, UNK)
    return arr

def feats(url: str):
    ext = tldextract.extract(url)
    dom = ext.top_domain_under_public_suffix or url
    L   = len(dom)
    f   = lambda c: dom.count(c)
    return [
        L,
        sum(1 for c in dom if not c.isalnum()),
        dom.count('-'),
        sum(c.isdigit() for c in dom),
        sum(c.isdigit() for c in dom) / L,
        len(ext.domain),
        int(ext.subdomain.count('.') + 1 if ext.subdomain else 0),
        -sum((f(c)/L)*math.log2(f(c)/L) for c in set(dom)),
    ]

# ────────────── main routine ────────────────────────────────────────── #
def main() -> None:
    df   = pd.read_csv(RAW)
    urls = df.url.values
    y    = df.label.values.astype(np.int32)

    X_num  = np.stack([feats(u)  for u in urls]).astype(np.float32)
    X_char = np.stack([encode(u) for u in urls])

    tr, va, te = group_split(urls, y)
    Xtr_num, Xva_num, Xte_num = X_num[tr], X_num[va], X_num[te]
    Xtr_ch,  Xva_ch,  Xte_ch  = X_char[tr], X_char[va], X_char[te]
    ytr, yva, yte             = y[tr], y[va], y[te]

    sc = StandardScaler().fit(Xtr_num)
    Xtr_num, Xva_num, Xte_num = map(sc.transform, (Xtr_num, Xva_num, Xte_num))

    np.savez(
        DATA / "dataset.npz",
        Xtr_num=Xtr_num, Xva_num=Xva_num, Xte_num=Xte_num,
        Xtr_ch=Xtr_ch,   Xva_ch=Xva_ch,   Xte_ch=Xte_ch,
        ytr=ytr, yva=yva, yte=yte,
    )
    json.dump(
        {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
        open(DATA / "scaler.json", "w"), indent=2
    )
    print("dataset.npz + scaler.json written")

if __name__ == "__main__":
    main()
