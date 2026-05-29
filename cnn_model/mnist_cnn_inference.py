"""
summit_inference_cnn.py
------------------------
Runs CNN TFLite inference on the Ripple Summit (i686, Python 3.7).

Copy these files to /var/rppl/storage/ on the Summit:
  mnist_cnn_model.tflite
  cnn_test_samples.npy
  cnn_test_labels.npy
  summit_inference_cnn.py

Then run:
  python37 /var/rppl/storage/summit_inference_cnn.py
"""

import sys
import time
import os

BASE         = "/var/rppl/storage"
MODEL_PATH   = os.path.join(BASE, "mnist_cnn_model.tflite")
SAMPLES_PATH = os.path.join(BASE, "cnn_test_samples.npy")
LABELS_PATH  = os.path.join(BASE, "cnn_test_labels.npy")

print("=" * 50)
print("Summit CNN TFLite Inference Test")
print("=" * 50)
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
    samples = np.load(SAMPLES_PATH)  # shape: (N, 28, 28, 1)
    labels  = np.load(LABELS_PATH)
    print(f"[OK] Loaded {len(samples)} test samples  shape: {samples.shape[1:]}")
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

# ── Run inference ────────────────────────────────────────────────────────────
print(f"\n── Running inference on {len(samples)} samples ──")

correct  = 0
times_ms = []

for i in range(len(samples)):
    x = samples[i : i + 1].astype(inp_details["dtype"])  # (1, 28, 28, 1)

    t0 = time.time()
    interpreter.set_tensor(inp_details["index"], x)
    interpreter.invoke()
    elapsed_ms = (time.time() - t0) * 1000
    times_ms.append(elapsed_ms)

    output = interpreter.get_tensor(out_details["index"])
    pred   = int(np.argmax(output))
    truth  = int(labels[i])
    status = "OK" if pred == truth else "WRONG"

    print(f"  Sample {i+1:2d}: predicted={pred}  truth={truth}  [{status}]  ({elapsed_ms:.1f} ms)")

    if pred == truth:
        correct += 1

# ── Summary ──────────────────────────────────────────────────────────────────
n = len(samples)
print()
print("=" * 50)
print(f"Accuracy        : {correct}/{n}  ({100*correct/n:.1f}%)")
print(f"Avg latency     : {sum(times_ms)/n:.1f} ms per inference")
print(f"Min latency     : {min(times_ms):.1f} ms")
print(f"Max latency     : {max(times_ms):.1f} ms")
print("=" * 50)

if correct == n:
    print("\nAll samples correct. CNN TFLite is working on the Summit.")
elif correct >= n * 0.8:
    print("\nMost samples correct. CNN inference is functional.")
else:
    print("\nLow accuracy — check that the model and test files match.")