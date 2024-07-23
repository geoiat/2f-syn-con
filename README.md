# 2f-syn-con

Code for the  manuscript
   > ***Two-factor synaptic consolidation reconciles robust memory with pruning and homeostatic scaling***<br>
     by Georgios Iatropoulos, Wulfram Gerstner, Johanni Brea (2024)

### Requirements
The main code is included in `RNNTorchTensor.py` and requires the following packages:
- PyTorch
- NumPy
- SciPy
- Scikit-learn

### Usage
To run the simulation presented in Figures 2 and 4, use
```
$ python RNN_50.py
```
The results are saved as a dictionary in the file `syn-con.pt`.

To run the simulation presented in Figure 3, use
```
$ python RNN_wake-sleep.py
```
The results are saved as a dictionary in the file `wake-sleep.pt`.

To run the simulation presented in Figures 5 and 6, use
```
$ python SYN-diff.py
```
The results are saved as a dictionary in the file `syn-diff.npy`.
