import pandas as pd
import numpy as np
import tldextract
import math
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).with_suffix("").parent / 'data'
df   = pd.read_csv(DATA / "raw.csv")

# 1) Compute features on URLs and keep raw strings
def feats(url: str):
    length  = len(url)
    specials = sum(1 for c in url if not c.isalnum())
    hyphens  = url.count('-')
    digits   = sum(c.isdigit() for c in url)
    ratio_d  = digits / length if length > 0 else 0
    ext = tldextract.extract(url)
    subdoms  = ext.subdomain.count('.') + 1 if ext.subdomain else 0
    dom_len  = len(ext.domain)
    puny     = url.startswith("http://xn--") or url.startswith("https://xn--")
    # Shannon entropy
    freq = [url.count(c) for c in set(url)]
    entropy = -sum((f/length) * math.log2(f/length) for f in freq) if length > 0 else 0
    return [length, specials, hyphens, digits, ratio_d,
            dom_len, subdoms, entropy, int(puny)]

# Gather raw URL strings and numeric features
urls = df.url.values
X = np.array([feats(u) for u in urls], dtype=np.float32)
y = df.label.values.astype(np.int32)

# Split into train/validation/test, preserving alignment of URLs
Xtr, Xtmp, ytr, ytmp, strs_tr, strs_tmp = train_test_split(
    X, y, urls,
    stratify=y,
    test_size=0.30,
    random_state=42
)
Xva, Xte, yva, yte, strs_va, strs_te = train_test_split(
    Xtmp, ytmp, strs_tmp,
    stratify=ytmp,
    test_size=0.50,
    random_state=42
)

# Standardize numeric features
sc = StandardScaler().fit(Xtr)
Xtr = sc.transform(Xtr)
Xva = sc.transform(Xva)
Xte = sc.transform(Xte)

# Save numeric and raw-string datasets
np.savez(
    DATA / "dataset.npz",
    X_train=Xtr,
    y_train=ytr,
    X_val=Xva,
    y_val=yva,
    X_test=Xte,
    y_test=yte,
    X_train_str=strs_tr,
    X_val_str=strs_va,
    X_test_str=strs_te
)

# Save scaler parameters
json.dump(
    {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
    open(DATA / 'scaler.json', 'w'),
    indent=2
)

print("dataset.npz + scaler.json written")