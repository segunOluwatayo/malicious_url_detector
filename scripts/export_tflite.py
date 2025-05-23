# scripts/export_tflite.py
import tensorflow as tf
from feeds import DATA

def main() -> None:
    model = tf.keras.models.load_model(DATA / "cnn.h5")
    conv  = tf.lite.TFLiteConverter.from_keras_model(model)
    # conv.optimizations = [tf.lite.Optimize.DEFAULT]
    (DATA / "url_cnn_fp32.tflite").write_bytes(conv.convert())
    print("url_cnn_int8.tflite exported")

if __name__ == "__main__":
    main()
