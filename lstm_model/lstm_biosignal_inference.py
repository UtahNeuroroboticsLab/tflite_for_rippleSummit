"""
summit_inference_lstm.py
-------------------------
Runs LSTM TFLite inference on the Ripple Summit (i686, Python 3.7).
Classifies 200ms windows of 4-channel biosignal data into movement intent
classes: rest / hand-open / hand-close / wrist-flex.

Copy these files to /var/rppl/storage/ on the Summit:
  lstm_biosignal_model.tflite
  lstm_test_windows.npy
  lstm_test_labels.npy
  summit_inference_lstm.py

Then run:
  python37 /var/rppl/storage/summit_inference_lstm.py
"""

import sys
import time
import os

BASE         = "/var/rppl/storage"
MODEL_PATH   = os.path.join(BASE, "lstm_biosignal_model.tflite")
WINDOWS_PATH = os.path.join(BASE, "lstm_test_windows.npy")
LABELS_PATH  = os.path.join(BASE, "lstm_test_labels.npy")

CLASS_NAMES = ["rest", "hand-open", "hand-close", "wrist-flex"]

print("=" * 55)
print("Summit LSTM Biosignal TFLite Inference Test")
print("=" * 55)
print(f"Python  : {sys.version}")
print(f"Model   : {MODEL_PATH}")
print()

# ── Imports ──────────────────────────────────────────────────────────────────
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

# ── Load test data ───────────────────────────────────────────────────────────
try:
    windows = np.load(WINDOWS_PATH)   # shape: (N, timesteps, channels)
    labels  = np.load(LABELS_PATH)
    n_samples, n_steps, n_ch = windows.shape
    print(f"[OK] Loaded {n_samples} windows  shape: ({n_steps} steps, {n_ch} channels)")
except Exception as e:
    print(f"[FAIL] Could not load test data: {e}")
    sys.exit(1)

# ── Load TFLite model ────────────────────────────────────────────────────────
try:
    t0 = time.time()
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    load_ms = (time.time() - t0) * 1000
    print(f"[OK] Model loaded in {load_ms:.1f} ms")
except Exception as e:
    print(f"[FAIL] Could not load model: {e}")
    sys.exit(1)

# ── Inspect model I/O ────────────────────────────────────────────────────────
inp_details = interpreter.get_input_details()[0]
out_details = interpreter.get_output_details()[0]
print(f"\nInput  shape : {inp_details['shape']}  dtype: {inp_details['dtype'].__name__}")
print(f"Output shape : {out_details['shape']}  dtype: {out_details['dtype'].__name__}")
print(f"Classes      : {CLASS_NAMES}")

# ── Run inference ────────────────────────────────────────────────────────────
print(f"\n── Running inference on {n_samples} windows ──")

correct  = 0
times_ms = []

for i in range(n_samples):
    x = windows[i : i + 1].astype(inp_details["dtype"])   # (1, timesteps, channels)

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
        f"  [{i+1:2d}] pred={CLASS_NAMES[pred]:<12s} "
        f"truth={CLASS_NAMES[truth]:<12s} "
        f"conf={conf:5.1f}%  [{status}]  ({elapsed_ms:.1f} ms)"
    )

    if pred == truth:
        correct += 1

# ── Per-class breakdown ──────────────────────────────────────────────────────
print()
print("── Per-class results ──")
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    cls_mask    = labels == cls_idx
    cls_total   = int(cls_mask.sum())
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
    print(f"  {cls_name:<14s}: {cls_correct}/{cls_total}")

# ── Summary ──────────────────────────────────────────────────────────────────
n = n_samples
print()
print("=" * 55)
print(f"Overall accuracy : {correct}/{n}  ({100*correct/n:.1f}%)")
print(f"Avg latency      : {sum(times_ms)/n:.1f} ms per window")
print(f"Min latency      : {min(times_ms):.1f} ms")
print(f"Max latency      : {max(times_ms):.1f} ms")
print(f"Throughput       : {1000/(sum(times_ms)/n):.1f} classifications/sec")
print("=" * 55)

# ── Real-time viability check ────────────────────────────────────────────────
avg_ms = sum(times_ms) / n
window_ms = n_steps  # 1 sample = 1 ms at 1000 Hz
print()
if avg_ms < window_ms * 0.5:
    print(f"Real-time verdict: FEASIBLE")
    print(f"  Inference ({avg_ms:.1f} ms) << window ({window_ms} ms)")
    print(f"  Plenty of headroom for real-time decoding at {window_ms}ms windows.")
elif avg_ms < window_ms:
    print(f"Real-time verdict: MARGINAL")
    print(f"  Inference ({avg_ms:.1f} ms) is close to window ({window_ms} ms).")
    print(f"  Consider reducing window overlap or model size.")
else:
    print(f"Real-time verdict: NOT FEASIBLE at {window_ms}ms windows")
    print(f"  Inference ({avg_ms:.1f} ms) > window ({window_ms} ms).")
    print(f"  Consider a larger window, fewer LSTM units, or quantised model.")