# scripts/test.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import tensorflow as tf
import pickle
import tldextract

from feeds import DATA
from build_dataset import encode, feats
from allowlist import GOOD

# ─── load assets ─────────────────────────────────────────────────────── #
model = tf.keras.models.load_model(DATA / "cnn.h5")
thr   = json.load(open(DATA / "threshold.json"))["threshold"]

# Load Bloom filter of known-bad domains
with open(DATA / "bad_domains.bloom", "rb") as f:
    bloom = pickle.load(f)

def predict(url: str) -> float:
    x_ch  = encode(url)[None, :]
    x_num = np.asarray([feats(url)], np.float32)
    return float(model.predict([x_ch, x_num], verbose=0)[0, 0])

def verdict(url: str) -> str:
    # Extract the registered domain
    dom = tldextract.extract(url).registered_domain

    # 1) Always-benign allow-list
    if dom in GOOD:
        return "Benign (allow-list)"

    # 2) If in bloom (feed-based malicious), trust the feed first
    if dom in bloom:
        score = predict(url)
        if score < thr:
            return f"Benign (bloom FP, p={score:.3f})"
        return "Malicious (feed)"

    # 3) Otherwise use the ML model
    score = predict(url)
    label = "Malicious" if score >= thr else "Benign"
    return f"{label} (p={score:.3f})"

# ─── demo urls ───────────────────────────────────────────────────────── #
tests = [
    "https://github.com",
    "https://google.com",
    "https://apple.com/store",
    "https://fantasticfilms.ru",
    "http://free-gift-cards.xyz",
    "http://amazonn-signin.com/login.php",
    "https://www.very.ie",
    "https://www.gardensrestaurantandcatering.com/",
    "https://www.scamadviser.com",
    "https://www.scamwatch.gov.au",
    "https://www.scamwatch.gov.au/report-a-scam",
    "https://anix.com.pl/",
    "https://tiffanycoshop.com",
    "https://willow.ie",
    "Luvasti.com"
]

for u in tests:
    print(f"{u:<45} -> {verdict(u)}")

f = feats("https://fantasticfilms.ru")
import json
sc = json.load(open("scripts/data/scaler.json"))
mean, scale = np.array(sc["mean"]), np.array(sc["scale"])
print("raw-feats:", f)
print("scaled:", (f - mean)/scale)
print("char-ids:", encode("https://fantasticfilms.ru")[:10])
url = "https://fantasticfilms.ru"
proba = predict(url)
print(f"{url} -> p={proba:.3f}")
from pybloom_live import BloomFilter
import pickle

# (Re)load your source list of malicious domains…
with open('source_bad_domains.txt') as f:
    domains = [d.strip() for d in f if d.strip()]

# Build the Bloom filter
bf = BloomFilter(capacity=len(domains), error_rate=0.001)
for d in domains:
    bf.add(d)

# Persist the bloom (for Python use, if you like)
with open('bad_domains.bloom', 'wb') as f:
    pickle.dump(bf, f)

# **Also** write out the text list you’ll ship to Android
with open('bad_domains.txt', 'w') as f:
    for d in domains:
        f.write(d + '\n')
import numpy as np

url = "https://fantasticfilms.ru"

# compute raw features once
raw_feats = feats(url)

# OPTION A: manual scaling
mean_  = np.array(sc["mean"])
scale_ = np.array(sc["scale"])
scaled_feats = (np.array(raw_feats) - mean_) / scale_

# OPTION B: using the scaler
# X = np.array(raw_feats).reshape(1, -1)
# scaled_feats = sc.transform(X).flatten()

print("── PYTHON DEBUG ──")
print("features :", raw_feats)
print("scaled   :", scaled_feats)
print("char ids :", encode(url)[:50])
print("model p  :", predict(url))
