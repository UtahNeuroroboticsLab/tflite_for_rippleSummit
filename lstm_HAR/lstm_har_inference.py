"""
summit_inference_har.py
------------------------
Runs HAR LSTM TFLite inference on the Ripple Summit (i686, Python 3.7).

Classifies 128-step (2.56s @ 50Hz) windows of 9-channel IMU sensor data
into 6 activity classes using a pre-trained LSTM.

Copy these files to /var/rppl/storage/ on the Summit:
  har_lstm_model.tflite
  har_test_windows.npy
  har_test_labels.npy
  summit_inference_har.py

Then run:
  python37 /var/rppl/storage/summit_inference_har.py
"""

import sys
import time
import os

BASE         = "/var/rppl/storage"
MODEL_PATH   = os.path.join(BASE, "har_lstm_model.tflite")
WINDOWS_PATH = os.path.join(BASE, "har_test_windows.npy")
LABELS_PATH  = os.path.join(BASE, "har_test_labels.npy")

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

print("=" * 60)
print("Summit HAR LSTM TFLite Inference Test")
print("UCI Human Activity Recognition Dataset")
print("=" * 60)
print(f"Python : {sys.version}")
print(f"Model  : {MODEL_PATH}")
print()

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    print("[OK] tflite_runtime imported")
except ImportError as e:
    print(f"[FAIL] Could not import tflite_runtime: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"[OK] numpy {np.__version__} imported")
except ImportError as e:
    print(f"[FAIL] Could not import numpy: {e}")
    sys.exit(1)

# ── Load test data ────────────────────────────────────────────────────────────
try:
    windows = np.load(WINDOWS_PATH)   # shape: (N, 128, 9)
    labels  = np.load(LABELS_PATH)
    n_samples, n_steps, n_ch = windows.shape
    window_ms = n_steps * (1000 / 50)   # 50Hz sampling rate
    print(f"[OK] Loaded {n_samples} windows  ({n_steps} steps, {n_ch} channels, {window_ms:.0f} ms each)")
except Exception as e:
    print(f"[FAIL] Could not load test data: {e}")
    sys.exit(1)

# ── Load TFLite model ─────────────────────────────────────────────────────────
try:
    t0 = time.time()
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    load_ms = (time.time() - t0) * 1000
    print(f"[OK] Model loaded in {load_ms:.1f} ms")
except Exception as e:
    print(f"[FAIL] Could not load model: {e}")
    sys.exit(1)

inp_details = interpreter.get_input_details()[0]
out_details = interpreter.get_output_details()[0]
print(f"\nInput  shape : {inp_details['shape']}  dtype: {inp_details['dtype'].__name__}")
print(f"Output shape : {out_details['shape']}  dtype: {out_details['dtype'].__name__}")
print(f"Activities   : {ACTIVITY_NAMES}")

# ── Run inference ─────────────────────────────────────────────────────────────
print(f"\n── Running inference on {n_samples} windows ──")

correct  = 0
times_ms = []

for i in range(n_samples):
    x = windows[i : i + 1].astype(inp_details["dtype"])

    t0 = time.time()
    interpreter.set_tensor(inp_details["index"], x)
    interpreter.invoke()
    elapsed_ms = (time.time() - t0) * 1000
    times_ms.append(elapsed_ms)

    probs  = interpreter.get_tensor(out_details["index"])[0]
    pred   = int(np.argmax(probs))
    truth  = int(labels[i])
    conf   = float(probs[pred]) * 100
    status = "OK" if pred == truth else "WRONG"

    print(
        f"  [{i+1:2d}] pred={ACTIVITY_NAMES[pred]:<22s} "
        f"truth={ACTIVITY_NAMES[truth]:<22s} "
        f"conf={conf:5.1f}%  [{status}]  ({elapsed_ms:.1f} ms)"
    )

    if pred == truth:
        correct += 1

# ── Per-class breakdown ───────────────────────────────────────────────────────
print()
print("── Per-class results ──")
for cls_idx, cls_name in enumerate(ACTIVITY_NAMES):
    cls_mask = labels == cls_idx
    cls_total = int(cls_mask.sum())
    if cls_total == 0:
        continue
    cls_windows = windows[cls_mask]
    cls_correct = 0
    for w in cls_windows:
        x = w[np.newaxis].astype(inp_details["dtype"])
        interpreter.set_tensor(inp_details["index"], x)
        interpreter.invoke()
        p = int(np.argmax(interpreter.get_tensor(out_details["index"])))
        if p == cls_idx:
            cls_correct += 1
    print(f"  {cls_name:<25s}: {cls_correct}/{cls_total}")

# ── Summary ───────────────────────────────────────────────────────────────────
n = n_samples
avg_ms = sum(times_ms) / n
print()
print("=" * 60)
print(f"Overall accuracy  : {correct}/{n}  ({100*correct/n:.1f}%)")
print(f"Expected benchmark: ~92% on full UCI HAR test set")
print(f"Avg latency       : {avg_ms:.1f} ms per window")
print(f"Min latency       : {min(times_ms):.1f} ms")
print(f"Max latency       : {max(times_ms):.1f} ms")
print(f"Throughput        : {1000/avg_ms:.1f} windows/sec")
print("=" * 60)

# ── Real-time viability ───────────────────────────────────────────────────────
print()
print("── Real-time viability ──")
print(f"  Window duration  : {window_ms:.0f} ms")
print(f"  Inference latency: {avg_ms:.1f} ms")
print(f"  Latency / window : {avg_ms/window_ms*100:.1f}%")
print()
if avg_ms < window_ms * 0.25:
    print("  FEASIBLE — inference uses <25% of window duration.")
    print("  Well within real-time budget even with 75% window overlap.")
elif avg_ms < window_ms * 0.5:
    print("  FEASIBLE — inference uses <50% of window duration.")
    print("  Real-time decoding viable with moderate overlap.")
elif avg_ms < window_ms:
    print("  MARGINAL — inference approaches window duration.")
    print("  Reduce window overlap or consider model quantisation.")
else:
    print("  NOT FEASIBLE at this window size.")
    print("  Consider a larger window or smaller LSTM.")