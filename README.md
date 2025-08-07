# 2f-syn-con

This repository contains simulation code for the  manuscript
   > [**Two-factor synaptic consolidation reconciles robust memory with pruning and homeostatic scaling**](https://doi.org/10.1101/2024.07.23.604787)<br>
     by Georgios Iatropoulos, Wulfram Gerstner, Johanni Brea (2024)

## Requirements
The scripts and notebooks are all written in `Python 3.x` and require the following packages:
- PyTorch (1.9.1 + CUDA 10.2)
- NumPy (1.21.2)
- SciPy (1.7.1)
- Scikit-learn (0.24.2)
- Matplotlib
- Jupyterlab

The packages can be installed all at once with:
```
$ pip install -r requirements.txt
```

## Usage
The main code to setup and train an RNN with the multi-factor synaptic consolidation model is located in `RNNTorchTensor.py`.

The `data` folder contains the results of the simulations as well as some experimental data.

The `theory` folder contains numerical solutions to the theoretical equations.

### *Figure 2*
To run the consolidation model in an RNN with balanced patterns, i.e. $f=0.5$, as presented in Figure 2, enter
```
$ python3 RNN_f50.py
```
The results are saved as a dictionary in the file `rnn-f50.pt`.
This can then be used in notebook `figure_2.ipynb` to reproduce the figure panels.

### *Figure 3*
To run the simulation of memory encoding and consolidation during wakefulness and sleep, as presented in Figure 3, enter
```
$ python3 RNN_wake-sleep.py
```
The results are saved as a dictionary in the file `rnn_wake-sleep.pt`.
This is used in notebook `figure_3.ipynb` to reproduce the figure panels.

### *Figures 5 & 6*
To run the simulation of synaptic intrinsic noise fluctuations, as presented in Figures 5 and 6, enter
```
$ python3 SYN_noise.py
```
The results are saved as a dictionary in the file `syn_noise.npy`.
This is used in notebook `figure_5-6.ipynb` to reproduce the figure panels.
