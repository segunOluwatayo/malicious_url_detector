import numpy as np, tensorflow as tf, json
from pathlib import Path
import tokenise as tok

DATA = Path(__file__).with_suffix('').parent / 'data'
ds   = np.load(DATA/'dataset.npz', allow_pickle=True)

# ── encode validation URLs ────────────────────────────────────────────
Xva = np.stack([tok.encode(u) for u in ds['X_val_str']], dtype=np.int32)
yva = ds['y_val'].astype(np.float32)

# ── get raw model probabilities ───────────────────────────────────────
model = tf.keras.models.load_model(DATA/'cnn.h5')
p = model.predict(Xva, verbose=0).flatten().astype(np.float32)

# clip to avoid log(0)
p = np.clip(p, 1e-6, 1.0 - 1e-6)

# logits = log(p / (1-p))
logit = np.log(p) - np.log1p(-p)

# convert to tensors once
logit_tf = tf.constant(logit, dtype=tf.float32)
yva_tf   = tf.constant(yva , dtype=tf.float32)

# ── temperature scalar we want to learn ───────────────────────────────
T = tf.Variable(1.0, dtype=tf.float32)
opt = tf.keras.optimizers.Adam(0.01)

for _ in range(500):
    with tf.GradientTape() as tape:
        # ensure tape tracks T
        scaled = tf.sigmoid(logit_tf / T)
        loss   = tf.keras.losses.binary_crossentropy(yva_tf, scaled)
    grad = tape.gradient(loss, T)
    opt.apply_gradients([(grad, T)])

json.dump({'T': float(T.numpy())}, open(DATA/'calibration.json','w'), indent=2)
print(f"calibration.json written (T ≈ {T.numpy():.3f})")
