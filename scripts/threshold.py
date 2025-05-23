from sklearn.metrics import roc_curve
import numpy as np, joblib, pathlib as p, json, tensorflow as tf
DATA = p.Path(__file__).with_suffix('').parent / 'data'
ds = np.load(DATA/'dataset.npz')
Xva,yva = ds['X_val'],ds['y_val']
proba   = tf.keras.models.load_model(DATA/'mlp.h5').predict(Xva).flatten()
fpr,tpr,thr = roc_curve(yva,proba)
best = thr[(tpr>=0.93)&(fpr<=0.05)][0]    # choose trade-off you like
json.dump({"mlp_threshold":float(best)}, open(DATA/'threshold.json','w'), indent=2)
print("best threshold=",best)
