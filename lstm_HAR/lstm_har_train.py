"""
train_har_lstm_tflite.py
-------------------------
Trains an LSTM on the UCI Human Activity Recognition (HAR) dataset and
exports to TFLite for validation on the Ripple Summit.

Dataset: UCI HAR Dataset (Anguita et al., 2013)
  - 30 subjects wearing a smartphone on the waist
  - 6 activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,
                  SITTING, STANDING, LAYING
  - 9 sensor channels: body_acc (xyz), body_gyro (xyz), total_acc (xyz)
  - 128-step windows @ 50Hz with 50% overlap
  - Well-established LSTM benchmark: expected accuracy ~92%

Why this dataset?
  Structure mirrors neural decoding: fixed-length windows of multi-channel
  sensor time series → discrete state classification. Swap the IMU channels
  for USEA electrode channels and you have the same pipeline.

Outputs:
  har_lstm_model.tflite   — model to copy to Summit
  har_test_windows.npy    — test windows  shape: (N, 128, 9)
  har_test_labels.npy     — ground-truth activity labels (0-indexed)

Run on Windows:
  pip install tensorflow numpy requests
  python train_har_lstm_tflite.py
"""

import os
import io
import zipfile
import urllib.request
import numpy as np
import tensorflow as tf

print(f"TensorFlow : {tf.__version__}")
print(f"NumPy      : {np.__version__}")

# ── Download dataset ─────────────────────────────────────────────────────────
DATASET_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATASET_DIR  = "UCI_HAR_Dataset"
DATASET_ZIP  = "UCI_HAR_Dataset.zip"

if not os.path.isdir(DATASET_DIR):
    if not os.path.isfile(DATASET_ZIP):
        print(f"\nDownloading UCI HAR dataset (~60 MB)...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_ZIP)
        print("Download complete.")
    print("Extracting...")
    with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
        zf.extractall(DATASET_DIR)
    print("Extracted.\n")
else:
    print(f"Dataset already present at ./{DATASET_DIR}/\n")

# ── Load raw inertial signals ────────────────────────────────────────────────
SIGNALS = [
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

BASE = os.path.join(DATASET_DIR, "UCI HAR Dataset")

def load_signals(split):
    """Load all 9 signal files for a given split (train/test).
    Returns array of shape (n_windows, 128, 9)."""
    channels = []
    for sig in SIGNALS:
        path = os.path.join(BASE, split, "Inertial Signals", f"{sig}_{split}.txt")
        data = np.loadtxt(path)   # shape: (n_windows, 128)
        channels.append(data)
    # Stack along channel axis → (n_windows, 128, 9)
    return np.stack(channels, axis=-1).astype(np.float32)

def load_labels(split):
    """Load labels for a split. Returns 0-indexed array."""
    path = os.path.join(BASE, split, f"y_{split}.txt")
    return np.loadtxt(path, dtype=np.int32) - 1   # convert 1-6 → 0-5

print("Loading training data...")
X_train = load_signals("train")
y_train = load_labels("train")

print("Loading test data...")
X_test  = load_signals("test")
y_test  = load_labels("test")

print(f"\nTrain : {X_train.shape}  labels: {np.bincount(y_train)}")
print(f"Test  : {X_test.shape}   labels: {np.bincount(y_test)}")
print(f"Window: {X_train.shape[1]} steps @ 50Hz = {X_train.shape[1]/50*1000:.0f} ms")
print(f"Channels: {SIGNALS}")

# ── Normalise ────────────────────────────────────────────────────────────────
# Per-channel normalisation across training set
mean = X_train.mean(axis=(0, 1), keepdims=True)
std  = X_train.std(axis=(0, 1),  keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

# ── Build LSTM model ─────────────────────────────────────────────────────────
N_TIMESTEPS = X_train.shape[1]   # 128
N_CHANNELS  = X_train.shape[2]   # 9
N_CLASSES   = len(ACTIVITY_NAMES) # 6

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(N_TIMESTEPS, N_CHANNELS)),

    tf.keras.layers.LSTM(128, return_sequences=True, unroll=True),
    tf.keras.layers.LSTM(64, unroll=True),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
], name="har_lstm")

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy : {acc:.4f}  (loss: {loss:.4f})")
print(f"Benchmark     : ~0.92 expected for this dataset with LSTM")

# ── Per-class accuracy ────────────────────────────────────────────────────────
print("\nPer-class accuracy:")
preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
for i, name in enumerate(ACTIVITY_NAMES):
    mask = y_test == i
    cls_acc = (preds[mask] == y_test[mask]).mean()
    print(f"  {name:<25s}: {cls_acc:.3f}  (n={mask.sum()})")

# ── Convert to TFLite ─────────────────────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("har_lstm_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"\nSaved har_lstm_model.tflite ({len(tflite_model) / 1024:.1f} KB)")

# ── Save test windows ─────────────────────────────────────────────────────────
# Pick a few from each class so all 6 activities are represented
N_PER_CLASS = 4
indices = []
for cls in range(N_CLASSES):
    cls_idx = np.where(y_test == cls)[0]
    indices.extend(np.random.choice(cls_idx, min(N_PER_CLASS, len(cls_idx)), replace=False))
indices = np.array(indices)
np.random.shuffle(indices)

test_windows = X_test[indices].astype(np.float32)
test_labels  = y_test[indices].astype(np.int32)

np.save("har_test_windows.npy", test_windows)
np.save("har_test_labels.npy",  test_labels)

print(f"Saved har_test_windows.npy  — shape {test_windows.shape}")
print(f"Saved har_test_labels.npy   — {[ACTIVITY_NAMES[l] for l in test_labels]}")

# ── Local TFLite sanity check ─────────────────────────────────────────────────
print("\n── Sanity check with TFLite interpreter (on this machine) ──")
interpreter = tf.lite.Interpreter(model_path="har_lstm_model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

correct = 0
for i in range(len(test_windows)):
    interpreter.set_tensor(inp["index"], test_windows[i : i + 1])
    interpreter.invoke()
    pred = int(np.argmax(interpreter.get_tensor(out["index"])))
    if pred == test_labels[i]:
        correct += 1

print(f"Local TFLite accuracy: {correct}/{len(test_windows)}")
print("\nFiles to copy to the Summit:")
print("  har_lstm_model.tflite")
print("  har_test_windows.npy")
print("  har_test_labels.npy")
print("  summit_inference_har.py")