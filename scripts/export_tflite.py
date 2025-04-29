import tensorflow as tf
import pathlib as p
from tensorflow.keras.models import load_model

DATA = p.Path(__file__).parents[1] / "data"
model = load_model(DATA / "url_model.h5")

# 1) FP32 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter.convert()
(DATA / "url_classifier.tflite").write_bytes(tflite_fp32)

# 2) INT8 quantized TFLite
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset():
    import numpy as np
    ds = np.load(DATA / "dataset.npz")["X_train"]
    for i in range(100):
        yield [ds[i : i+1]]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_int8 = converter.convert()
(DATA / "url_classifier_int8.tflite").write_bytes(tflite_int8)

print("TFLite models written to:", 
      DATA / "url_classifier.tflite", 
      DATA / "url_classifier_int8.tflite")
