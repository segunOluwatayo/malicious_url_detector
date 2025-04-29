import numpy as np
import pandas as pd
import json
import pathlib as p
import tldextract
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
DATA = p.Path(__file__).parents[1] / "data"
raw_csv = DATA / "raw.csv"

# 1) Load raw URLs + labels
df = pd.read_csv(raw_csv)
urls = df["url"].tolist()
labels = df["label"].values

# 2) Feature extraction
def extract_features(url: str):
    length = len(url)
    specials = sum(1 for c in url if not c.isalnum())
    ext = tldextract.extract(url)
    subdomains = ext.subdomain.count(".") + 1 if ext.subdomain else 0
    # Shannon entropy of the whole URL string
    freqs = [url.count(ch) for ch in set(url)]
    entropy = -sum((f/length) * math.log2(f/length) for f in freqs if f>0)
    return [length, specials, subdomains, entropy]

# Build feature matrix
X = np.array([extract_features(u) for u in urls], dtype=np.float32)
y = labels.astype(np.int32)

# 3) Train/val/test split (70/15/15) stratified
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# 4) Scale numeric features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# 5) Save .npz and scaler.json
np.savez(
    DATA / "dataset.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test
)
with open(DATA / "scaler.json", "w") as f:
    json.dump({
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }, f, indent=2)

print("Saved dataset.npz and scaler.json:")
print("  • X_train:", X_train.shape, " y_train:", y_train.shape)
print("  • X_val:  ", X_val.shape,   " y_val:",   y_val.shape)
print("  • X_test: ", X_test.shape,  " y_test:",  y_test.shape)
