# # scripts/export_tflite.py
# import tensorflow as tf
# from feeds import DATA

# def main() -> None:
#     model = tf.keras.models.load_model(DATA / "cnn.h5")

#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     # ⇩  delete the two quantization lines
#     # converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     # converter.representative_dataset = ...

#     tflite_bytes = converter.convert()
#     (DATA / "url_cnn_fp32.tflite").write_bytes(tflite_bytes)
#     print("url_cnn_fp32.tflite exported")

# if __name__ == "__main__":
#     main()
# scripts/export_tflite.py
import tensorflow as tf
from feeds import DATA

def main() -> None:
    model = tf.keras.models.load_model(DATA / "cnn.h5")
    
    # Print model input details for debugging
    print("Original model inputs:")
    for i, input_layer in enumerate(model.inputs):
        print(f"Input {i}: {input_layer.name}, shape={input_layer.shape}, dtype={input_layer.dtype}")
    
    # Create the converter with specific settings
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Ensure float32 precision is maintained
    converter.target_spec.supported_types = [tf.float32]
    
    # Optional: Set optimization flags if needed
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = ...
    
    # Convert the model
    tflite_bytes = converter.convert()
    
    # Save the model
    output_path = DATA / "url_cnn_fp32.tflite"
    output_path.write_bytes(tflite_bytes)
    print(f"Model exported to {output_path}")
    
    # Optional: Inspect the converted model
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    print("\nTFLite model inputs:")
    for i, input_detail in enumerate(input_details):
        print(f"Input {i}: {input_detail['name']}, shape={input_detail['shape']}, dtype={input_detail['dtype']}")

if __name__ == "__main__":
    main()
