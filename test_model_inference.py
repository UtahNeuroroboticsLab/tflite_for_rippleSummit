import numpy as np
import tflite_runtime.interpreter as tflite

# 1. Load the model and allocate hardware buffers
model_path = "/var/rppl/storage/wheels/toy_model.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 2. Map out the input and output structural expectations
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Create a test input value (Let's pass the number 10.0)
test_input = np.array([[10.0]], dtype=np.float32)

# 4. Inject the data into the model's input memory allocation
interpreter.set_tensor(input_details[0]['index'], test_input)

# 5. Run the mathematical processing steps
interpreter.invoke()

# 6. Extract the processed calculation out of the output layer
raw_prediction = interpreter.get_tensor(output_details[0]['index'])

print("--- TF LITE TEST SUCCESSFUL ---")
print("Input passed to model:  10.0")
print("Predicted Output (X*3):", raw_prediction[0][0])