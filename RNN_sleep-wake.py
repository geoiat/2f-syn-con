#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Georgios Iatropoulos
"""

import torch
from RNNTorchTensors import ContinualRNN

# Set device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Set parameters
N = 1000
n_seeds = 1
f = 0.05
beta_vec = torch.tensor([20], device=device)
a_vec = torch.tensor([0.5, 0.2], device=device)
seed_vec = torch.arange(700, 700+n_seeds, device=device)

weight_type = 'hadamard'
mem_type = 'attractor'

n_a = a_vec.shape[0]

# Storage of all recorders
rec_dict = {}

# Run simulations
for i in range(n_a):
    seed = seed_vec[0].item()
    beta = beta_vec[0].item()
    a = a_vec[i].item()
    M = round(a*N)
    print(f'seed: {seed}, f: {f}, M: {M}, exact_mode: OFF, beta: {beta}')

    rnn = ContinualRNN(N=N, f=f, inh_ratio=0., z=2, u0=.6, theta0=1., length=70.,
                            eta_wake=1e-2, eta_sleep=1e-2, eta_theta=1e-2, seed=seed,
                            M_day=M, n_days=1, n_epochs_wake=35, n_epochs_sleep=100000,
                            norm_type='u', weight_type=weight_type, optimizer_type='bp', pattern_type='exact', mem_type=mem_type,
                            inh_type='unbalanced', with_bias=True, exact_mode=False, beta=beta, kernel_degree=1, record_on=True)
    rnn.init_patterns()
    rnn.init_tensors()
    rnn.init_recorders()
    rnn.optimize()

    rec_dict[M] = rnn.rec

# Save data
data = {}
data['date'] = rnn.get_date()
data['params'] = rnn.params
data['a'] = a_vec
data['beta'] = beta_vec
data['rec_dict'] = rec_dict
data['ksi_fam'] = rnn.X

torch.save(data, './data/rnn_wake-sleep.pt')
