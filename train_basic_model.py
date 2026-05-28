import tensorflow as tf
import numpy as np

# Define a super simple structure: Y = 3 * X
initializer = tf.keras.initializers.Constant([[3.0]])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], kernel_initializer=initializer, use_bias=False)
])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Convert it to the ultra-lightweight TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save it to your disk
with open("toy_model.tflite", "wb") as f:
    f.write(tflite_model)
print("toy_model.tflite successfully created!")