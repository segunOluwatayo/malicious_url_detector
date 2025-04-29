# scripts/train.py

import numpy as np
import pathlib as p
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Paths
DATA = p.Path(__file__).parents[1] / "data"
ds = np.load(DATA / "dataset.npz")
X_train, y_train = ds["X_train"], ds["y_train"]
X_val,   y_val   = ds["X_val"],   ds["y_val"]

# 1) Build model
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

# 2) Compile
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[tf.keras.metrics.Recall(name="recall")]
)

# 3) Callbacks
es = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# 4) Class weights to compensate imbalance
pos = sum(y_train)
neg = len(y_train) - pos
class_weight = {0: 1.0, 1: neg/pos}

# 5) Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[es],
    verbose=2
)

# 6) Save model
model.save(DATA / "url_model.h5")
print("Model saved to", DATA / "url_model.h5")
