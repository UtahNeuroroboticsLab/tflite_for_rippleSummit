"""
train_mnist_cnn_tflite.py
--------------------------
Trains a small CNN on MNIST, converts to TFLite, and saves:
  - mnist_cnn_model.tflite  : the model to copy to the Summit
  - cnn_test_samples.npy    : 20 test images (28x28x1 float32)
  - cnn_test_labels.npy     : ground-truth labels for those 20 images

Run on Windows:
  pip install tensorflow numpy
  python train_mnist_cnn_tflite.py
"""

import numpy as np
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

# ── Load MNIST ──────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalise to [0, 1] and add channel dimension: (N, 28, 28) → (N, 28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = x_train[..., np.newaxis]
x_test  = x_test[...,  np.newaxis]

print(f"Training samples : {len(x_train)}  shape: {x_train.shape[1:]}")
print(f"Test samples     : {len(x_test)}")

# ── Build CNN ────────────────────────────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10,  activation="softmax"),
], name="mnist_cnn")

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Train ────────────────────────────────────────────────────────────────────
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.4f}  (loss: {loss:.4f})")

# ── Convert to TFLite ────────────────────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("mnist_cnn_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Saved mnist_cnn_model.tflite ({len(tflite_model) / 1024:.1f} KB)")

# ── Save 20 test samples for Summit validation ───────────────────────────────
N = 20
indices = np.random.choice(len(x_test), N, replace=False)
samples = x_test[indices].astype("float32")   # shape: (20, 28, 28, 1)
labels  = y_test[indices].astype("int32")

np.save("cnn_test_samples.npy", samples)
np.save("cnn_test_labels.npy",  labels)

print(f"Saved cnn_test_samples.npy  — shape {samples.shape}")
print(f"Saved cnn_test_labels.npy   — labels: {labels.tolist()}")

# ── Quick sanity check with TFLite interpreter on this machine ───────────────
print("\n── Sanity check with TFLite interpreter (on this machine) ──")
interpreter = tf.lite.Interpreter(model_path="mnist_cnn_model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

correct = 0
for i in range(N):
    interpreter.set_tensor(inp["index"], samples[i : i + 1])
    interpreter.invoke()
    pred = int(np.argmax(interpreter.get_tensor(out["index"])))
    if pred == labels[i]:
        correct += 1

print(f"Local TFLite accuracy on {N} samples: {correct}/{N}")
print("\nFiles to copy to the Summit:")
print("  mnist_cnn_model.tflite")
print("  cnn_test_samples.npy")
print("  cnn_test_labels.npy")
print("  summit_inference_cnn.py")