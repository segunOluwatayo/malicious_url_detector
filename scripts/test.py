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
]

for u in tests:
    print(f"{u:<45} -> {verdict(u)}")
