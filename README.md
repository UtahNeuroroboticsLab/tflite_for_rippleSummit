# tflite_for_rippleSummit
tflite compiled to run on a Ripple Summit (i686 architecture)

## Info

Tested on a Ripple Summit running i686 GNU/Linux with an Intel(R) Celeron(R) CPU N2930 processor.

You do need to install Python 3.7 on the Ripple Summit as that is the earliest version that will work with this build.

Contact Leonardo.Ferrisi@utah.edu for information on how to do this if you run into issues.

## Setup

1.  Clone this repo

```bash
git clone git@github.com:UtahNeuroroboticsLab/tflite_for_rippleSummit.git
```

2. Navigate in and use your python installtion to install this whl file as a package
```bash
cd tflite_for_rippleSummit
python3 -m pip install tflite_runtime-2.5.0-cp37-cp37m-linux_i686.whl
```

3. Test out 

This is a very basic Neural Network. Just does X -> 3X with one layer.

Run the inference script on the Ripple Summit to confirm that it works.

```bash
python test_model_inference.py
```

Your expect output should be: 
```bash
--- TF LITE TEST SUCCESSFUL ---
Input passed to model:  10.0
Predicted Output (X*3):, 3.0
```