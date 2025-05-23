# scripts/train_model.py
"""Train the hybrid Char-CNN + numeric model and save cnn.h5."""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silence TF dll warnings
import numpy as np, tensorflow as tf, json
from pathlib import Path
from feeds import DATA

def build() -> tf.keras.Model:
    VOCAB_SIZE = 2 + 95
    INPUT_LEN  = 200
    NUM_FEATS  = 8

    ch_in  = tf.keras.layers.Input(shape=(INPUT_LEN,), dtype="int32", name="ch")
    num_in = tf.keras.layers.Input(shape=(NUM_FEATS,),                 name="num")

    x = tf.keras.layers.Embedding(VOCAB_SIZE, 32)(ch_in)
    x = tf.keras.layers.Conv1D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    y = tf.keras.layers.Dense(32, activation="relu")(num_in)
    z = tf.keras.layers.concatenate([x, y])
    z = tf.keras.layers.Dense(64, activation="relu")(z)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(z)
    return tf.keras.Model([ch_in, num_in], out)

def main() -> None:
    ds = np.load(DATA / "dataset.npz", allow_pickle=True)
    Xtr_ch, Xtr_num, ytr = ds["Xtr_ch"], ds["Xtr_num"], ds["ytr"]
    Xva_ch, Xva_num, yva = ds["Xva_ch"], ds["Xva_num"], ds["yva"]

    model = build()
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(5e-4),
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="rec")]
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )
    hist = model.fit(
        [Xtr_ch, Xtr_num], ytr,
        validation_data=([Xva_ch, Xva_num], yva),
        epochs=25, batch_size=256, callbacks=[es], verbose=2
    )

    model.save(DATA / "cnn.h5")
    json.dump(hist.history, open(DATA / "history.json", "w"))
    print("cnn.h5 saved")

if __name__ == "__main__":
    main()
