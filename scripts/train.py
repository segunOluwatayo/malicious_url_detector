import tensorflow as tf, numpy as np, pathlib as p
DATA = p.Path(__file__).with_suffix('').parent / 'data'
ds = np.load(DATA/"dataset.npz")
Xtr, ytr, Xva, yva = ds["X_train"],ds["y_train"],ds["X_val"],ds["y_val"]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xtr.shape[1],)),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=[tf.keras.metrics.Recall(name='recall')])
model.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=10,batch_size=1024,verbose=2)
model.save(DATA/'mlp.h5')
