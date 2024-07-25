# 2f-syn-con

This repository contains simulation code for the  manuscript
   > [**Two-factor synaptic consolidation reconciles robust memory with pruning and homeostatic scaling**](https://doi.org/10.1101/2024.07.23.604787)<br>
     by Georgios Iatropoulos, Wulfram Gerstner, Johanni Brea (2024)

### Requirements
The main code is included in `RNNTorchTensor.py` and uses the following packages:
- PyTorch (1.9.1)
- NumPy (1.21.2)
- SciPy (1.7.1)
- Scikit-learn (0.24.2)

### Usage
To run the simulation presented in Figures 2 and 4, use
```
$ python3 RNN_50.py
```
The results are saved as a dictionary in the file `syn-con.pt`.

To run the simulation presented in Figure 3, use
```
$ python3 RNN_wake-sleep.py
```
The results are saved as a dictionary in the file `wake-sleep.pt`.

To run the simulation presented in Figures 5 and 6, use
```
$ python3 SYN_diff.py
```
The results are saved as a dictionary in the file `syn-diff.npy`.
