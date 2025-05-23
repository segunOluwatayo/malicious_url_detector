import tensorflow as tf
from pathlib import Path
DATA = Path(__file__).with_suffix('').parent / 'data'

cnn = tf.keras.models.load_model(DATA/'cnn.h5')
conv = tf.lite.TFLiteConverter.from_keras_model(cnn)
conv.optimizations = [tf.lite.Optimize.DEFAULT]    # weight-only INT8
tflite = conv.convert()
open(DATA/'url_cnn_int8.tflite','wb').write(tflite)
print("url_cnn_int8.tflite exported (%d bytes)" % len(tflite))
