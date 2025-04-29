import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1) Load test data
ds = np.load("data/dataset.npz")
X_test, y_test = ds["X_test"], ds["y_test"]

# 2) Load your trained model
model = load_model("data/url_model.h5")

# 3) Get predictions
y_proba = model.predict(X_test, batch_size=256).flatten()
y_pred = (y_proba >= 0.5).astype(int)

# 4) Compute metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
