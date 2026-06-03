"""
train_biosignal_lstm_tflite.py
-------------------------------
Generates synthetic multi-channel biosignal data resembling neural LFP
(Local Field Potential) recordings, trains a small LSTM classifier to
decode "movement intent" classes from sliding windows, and exports to TFLite.

Why synthetic LFP?
  Real neural recordings during movement show distinct oscillatory patterns
  across frequency bands (theta ~8Hz, alpha ~12Hz, beta ~25Hz, gamma ~60Hz).
  This generator mimics that structure so the validation is meaningful for
  the actual USEA decoding task.

Outputs:
  lstm_biosignal_model.tflite  — model to copy to Summit
  lstm_test_windows.npy        — test windows  shape: (N, timesteps, channels)
  lstm_test_labels.npy         — ground-truth class labels

Run on Windows:
  pip install tensorflow numpy
  python train_biosignal_lstm_tflite.py
"""

import numpy as np
import tensorflow as tf

print(f"TensorFlow  : {tf.__version__}")
print(f"NumPy       : {np.__version__}")

# ── Signal parameters ────────────────────────────────────────────────────────
SEED         = 42
FS           = 1000        # sampling rate (Hz) — typical for neural recordings
WINDOW_MS    = 200         # classification window length (ms)
WINDOW_STEPS = int(FS * WINDOW_MS / 1000)   # 200 samples per window
N_CHANNELS   = 4           # simulated electrode channels
N_CLASSES    = 4           # rest / hand-open / hand-close / wrist-flex
N_TRAIN      = 2000        # windows per class in training set
N_TEST       = 200         # windows per class in test set

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Synthetic LFP generator ──────────────────────────────────────────────────
# Each class has a distinct dominant oscillation band that modulates amplitude,
# mirroring how motor cortex LFP shifts from beta (rest) to gamma (movement).
#
# Class 0 — Rest        : strong beta (25 Hz), weak gamma
# Class 1 — Hand open   : suppressed beta, strong gamma (60 Hz)
# Class 2 — Hand close  : strong alpha (12 Hz) + gamma
# Class 3 — Wrist flex  : strong theta (8 Hz) + alpha

CLASS_PROFILES = {
    0: {"freqs": [25],     "amps": [1.2],        "label": "rest"},
    1: {"freqs": [60],     "amps": [1.0],        "label": "hand-open"},
    2: {"freqs": [12, 60], "amps": [0.9, 0.7],   "label": "hand-close"},
    3: {"freqs": [8,  12], "amps": [1.0, 0.8],   "label": "wrist-flex"},
}

def generate_windows(n_per_class, n_steps, n_ch, fs):
    t = np.linspace(0, n_steps / fs, n_steps, endpoint=False)
    X, y = [], []
    for cls, profile in CLASS_PROFILES.items():
        for _ in range(n_per_class):
            # Build multi-channel window
            window = np.zeros((n_steps, n_ch), dtype=np.float32)
            for ch in range(n_ch):
                sig = np.zeros(n_steps)
                # Add class-defining oscillations with per-channel phase jitter
                for freq, amp in zip(profile["freqs"], profile["amps"]):
                    phase = np.random.uniform(0, 2 * np.pi)
                    ch_amp = amp * np.random.uniform(0.8, 1.2)
                    sig += ch_amp * np.sin(2 * np.pi * freq * t + phase)
                # Add broadband noise (mimics background neural activity)
                sig += np.random.normal(0, 0.5, n_steps)
                # Normalise channel
                sig = (sig - sig.mean()) / (sig.std() + 1e-8)
                window[:, ch] = sig.astype(np.float32)
            X.append(window)
            y.append(cls)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

print(f"\nGenerating synthetic LFP data...")
print(f"  Window    : {WINDOW_MS} ms  ({WINDOW_STEPS} samples @ {FS} Hz)")
print(f"  Channels  : {N_CHANNELS}")
print(f"  Classes   : {N_CLASSES}  {[v['label'] for v in CLASS_PROFILES.values()]}")

X_train, y_train = generate_windows(N_TRAIN, WINDOW_STEPS, N_CHANNELS, FS)
X_test,  y_test  = generate_windows(N_TEST,  WINDOW_STEPS, N_CHANNELS, FS)

print(f"  Train     : {X_train.shape}  labels: {np.bincount(y_train)}")
print(f"  Test      : {X_test.shape}   labels: {np.bincount(y_test)}")

# ── Build LSTM model ─────────────────────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_STEPS, N_CHANNELS)),

    # unroll=True eliminates dynamic TensorListReserve ops that break TFLite conversion.
    # Required for standard tflite_runtime (no SELECT_TF_OPS dependency on the Summit).
    tf.keras.layers.LSTM(64, return_sequences=True, unroll=True),
    tf.keras.layers.LSTM(32, unroll=True),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
], name="biosignal_lstm")

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Train ────────────────────────────────────────────────────────────────────
print("\nTraining...")
model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.4f}  (loss: {loss:.4f})")

# ── Convert to TFLite ────────────────────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("lstm_biosignal_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Saved lstm_biosignal_model.tflite ({len(tflite_model) / 1024:.1f} KB)")

# ── Save test windows for Summit ─────────────────────────────────────────────
N = 20
indices = np.random.choice(len(X_test), N, replace=False)
test_windows = X_test[indices]
test_labels  = y_test[indices]

np.save("lstm_test_windows.npy", test_windows)
np.save("lstm_test_labels.npy",  test_labels)

print(f"Saved lstm_test_windows.npy  — shape {test_windows.shape}")
print(f"Saved lstm_test_labels.npy   — labels: {test_labels.tolist()}")
class_names = [CLASS_PROFILES[i]["label"] for i in test_labels]
print(f"                               classes: {class_names}")

# ── Local TFLite sanity check ─────────────────────────────────────────────────
print("\n── Sanity check with TFLite interpreter (on this machine) ──")
interpreter = tf.lite.Interpreter(model_path="lstm_biosignal_model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

correct = 0
for i in range(N):
    interpreter.set_tensor(inp["index"], test_windows[i : i + 1])
    interpreter.invoke()
    pred = int(np.argmax(interpreter.get_tensor(out["index"])))
    if pred == test_labels[i]:
        correct += 1

print(f"Local TFLite accuracy on {N} samples: {correct}/{N}")
print("\nFiles to copy to the Summit:")
print("  lstm_biosignal_model.tflite")
print("  lstm_test_windows.npy")
print("  lstm_test_labels.npy")
print("  summit_inference_lstm.py")