# scripts/find_threshold.py
"""
Pick a threshold with ≈2% false-positive rate on the validation split.
"""

import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve
from feeds import DATA

TARGET_FPR = 0.02          # 2% max false positives on benign URLs

def main() -> None:
    # Load validation data
    ds = np.load(DATA / "dataset.npz", allow_pickle=True)
    Xva_ch, Xva_num, yva = ds["Xva_ch"], ds["Xva_num"], ds["yva"]

    # Load model & get predicted probabilities
    model = tf.keras.models.load_model(DATA / "cnn.h5")
    proba = model.predict([Xva_ch, Xva_num], verbose=0).flatten()

    # Compute ROC curve
    fpr, tpr, thr = roc_curve(yva, proba)

    # Consider only thresholds in [0,1]
    mask = np.isfinite(thr) & (thr >= 0.0) & (thr <= 1.0)
    fpr, thr = fpr[mask], thr[mask]

    # Pick the *largest* threshold whose FPR ≤ TARGET_FPR
    valid = thr[fpr <= TARGET_FPR]
    if valid.size > 0:
        best = float(valid.min())
    else:
    # use the 95th percentile of benign-probabilities
        benign_scores = proba[yva == 0]
        best = float(np.percentile(benign_scores, 95))
        print(f"Warning: no thr meets FPR≤{TARGET_FPR:.2%}; "
            f"falling back to 95th pct of benign = {best:.4f}")

    # Save threshold
    out = {"threshold": best}
    with open(DATA / "threshold.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"threshold.json saved  (thr = {best:.4f})")

if __name__ == "__main__":
    main()
