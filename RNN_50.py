#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Georgios Iatropoulos
"""


import torch
from RNNTorchTensors import RNN

# Set device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Set parameters
N = 1000
n_tests = 20
n_seeds = 1

weight_type = 'hadamard'
pattern_type = 'bernoulli'

# Vectors
seed_vec = torch.arange(100, 100+10*n_seeds, 10, device=device)
z_vec = torch.tensor([1, 2, 3, 4], device=device)
f_vec = torch.tensor([0.50], device=device)
a_vec = torch.cat( (torch.linspace(0.08, 0.33, 5, device=device), torch.linspace(0.33, 0.7, 4, device=device)[1:]) )
p_vec = torch.linspace(0, 500-10, 200, device=device)/N

# Hyperparameters
eta_vec = torch.tensor([1e-4, 5e-3, 7e-3, 7e-3], device=device)
eta_th_vec = torch.tensor([1e-3, 5e-3, 7e-3, 7e-3], device=device)
length_vec = torch.tensor([10, 20, 50, 50], device=device)
epochs_vec = torch.tensor([5, 6, 5, 5], device=device)*5

# Vector lengths
n_z = z_vec.shape[0]
n_f = f_vec.shape[0]
n_a = a_vec.shape[0]
n_p = p_vec.shape[0]

# Storage of stats
min_margin_mean = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
min_margin_std = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
avg_margin_mean = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
avg_margin_std = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
rho_mean = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
rho_std = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
L23_margin_mean = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
L23_margin_std = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
L24_margin_mean = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
L24_margin_std = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
syn_margin_mean = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
syn_margin_std = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
l0_mean = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
l0_std = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
l1_mean = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
l1_std = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
l2_mean = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
l2_std = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
sym_mat = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
theta_mat = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
bias_mat = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
conv_mat = torch.zeros((n_z,n_f,n_a,n_seeds), device=device)
w_mat = torch.zeros((n_z,N**2), device=device)
u_mat = torch.zeros((n_z,N**2), device=device)

# Storage of tolerance stats
cap_mat_iso = torch.zeros((n_z,n_f,n_a,n_seeds,n_p), device=device)
err_mat_iso = torch.zeros((n_z,n_f,n_a,n_seeds,n_p), device=device)
cap_mat_bal = torch.zeros((n_z,n_f,n_a,n_seeds,n_p), device=device)
err_mat_bal = torch.zeros((n_z,n_f,n_a,n_seeds,n_p), device=device)
cap_mat_syn_z = torch.zeros((n_z,n_f,n_a,n_seeds,n_p), device=device)
err_mat_syn_z = torch.zeros((n_z,n_f,n_a,n_seeds,n_p), device=device)
cap_mat_syn_1 = torch.zeros((n_z,n_f,n_a,n_seeds,n_p), device=device)
err_mat_syn_1 = torch.zeros((n_z,n_f,n_a,n_seeds,n_p), device=device)
cap_mat_syn_1mask = torch.zeros((n_z,n_f,n_a,n_seeds,n_p), device=device)
err_mat_syn_1mask = torch.zeros((n_z,n_f,n_a,n_seeds,n_p), device=device)

# Iterator for all indices
I = torch.cartesian_prod(torch.arange(n_z, device=device),
                         torch.arange(n_f, device=device),
                         torch.arange(n_a, device=device),
                         torch.arange(n_seeds, device=device))

# Run simulations
for i_z, i_f, i_a, i_s in I:
    z = z_vec[i_z].item()
    f = f_vec[i_f].item()
    a = a_vec[i_a].item()
    seed = seed_vec[i_s].item()
    
    print(f'seed: {seed}, f: {f}, alpha: {a}, z: {z}, weights: {weight_type}')

    M = round(a*N)

    rnn = RNN(N=N, M=M, f=f, inh_ratio=0., z=z, u0=0.1, theta0=0., eta=eta_vec[i_z].item(), eta_theta=eta_th_vec[i_z].item(), length=length_vec[i_z].item(), seed=seed,
              n_epochs=500*M*epochs_vec[i_z].item(), norm_type='u', weight_type=weight_type, optimizer_type='bp', pattern_type=pattern_type,
              inh_type='unbalanced', with_bias=True, kernel_degree=1, record_on=False)
    rnn.init_patterns()
    rnn.init_tensors()
    #rnn.init_recorders()
    rnn.optimize()

    print(f'margin: {rnn.get_margin()[0].item()}, rho: {rnn.get_rho()[0].item()}, l0: {rnn.get_l0()[0].item()}, conv: {rnn.conv_time}')
    
    rnn.scale_weights_to_ref('mean-u')

    min_margin_mean[i_z,i_f,i_a,i_s], min_margin_std[i_z,i_f,i_a,i_s] = rnn.get_min_margin()
    avg_margin_mean[i_z,i_f,i_a,i_s], avg_margin_std[i_z,i_f,i_a,i_s] = rnn.get_avg_margin()
    rho_mean[i_z,i_f,i_a,i_s], rho_std[i_z,i_f,i_a,i_s] = rnn.get_rho()
    L23_margin_mean[i_z,i_f,i_a,i_s], L23_margin_std[i_z,i_f,i_a,i_s] = rnn.get_norm_margin(2/3)
    L24_margin_mean[i_z,i_f,i_a,i_s], L24_margin_std[i_z,i_f,i_a,i_s] = rnn.get_norm_margin(2/4)
    l0_mean[i_z,i_f,i_a,i_s], l0_std[i_z,i_f,i_a,i_s] = rnn.get_l0()
    l1_mean[i_z,i_f,i_a,i_s], l1_std[i_z,i_f,i_a,i_s] = rnn.get_l1()
    l2_mean[i_z,i_f,i_a,i_s], l2_std[i_z,i_f,i_a,i_s] = rnn.get_l2()
    sym_mat[i_z,i_f,i_a,i_s] = rnn.get_symmetry()
    theta_mat[i_z,i_f,i_a,i_s] = rnn.get_theta()
    bias_mat[i_z,i_f,i_a,i_s] = rnn.get_bias().mean()
    conv_mat[i_z,i_f,i_a,i_s] = rnn.conv_time
    
    syn_margin_mean[i_z,i_f,i_a,i_s], syn_margin_std[i_z,i_f,i_a,i_s] = rnn.get_syn_margin()

    if i_f==0 and i_s==0 and i_a==4:
        w_mat[i_z,:] = rnn.W.flatten().clone()
        u_mat[i_z,:] = rnn.U[:,:,0].flatten().clone()

    # run robustness test
    for i_p in range(n_p):
        p = p_vec[i_p].item()
        rnn.dynamic_inh = False
        cap, _ = rnn.get_tolerance('bernoulli', p, n_tests=n_tests, n_steps=50)
        cap_mat_iso[i_z,i_f,i_a,i_s,i_p] = cap
        _, err = rnn.get_tolerance('bernoulli', p, n_tests=n_tests, n_steps=1)
        err_mat_iso[i_z,i_f,i_a,i_s,i_p] = err

        cap, _ = rnn.get_tolerance('bernoulli-balanced', p, n_tests=n_tests, n_steps=50)
        cap_mat_bal[i_z,i_f,i_a,i_s,i_p] = cap
        _, err = rnn.get_tolerance('bernoulli-balanced', p, n_tests=n_tests, n_steps=1)
        err_mat_bal[i_z,i_f,i_a,i_s,i_p] = err

        cap, _ = rnn.get_tolerance('gauss', p*0.1, n_tests=n_tests, n_steps=50)
        cap_mat_syn_z[i_z,i_f,i_a,i_s,i_p] = cap
        _, err = rnn.get_tolerance('gauss', p*0.1, n_tests=n_tests, n_steps=1)
        err_mat_syn_z[i_z,i_f,i_a,i_s,i_p] = err

        cap, _ = rnn.get_tolerance('gauss1', p*0.16, n_tests=n_tests, n_steps=50)
        cap_mat_syn_1[i_z,i_f,i_a,i_s,i_p] = cap
        _, err = rnn.get_tolerance('gauss1', p*0.16, n_tests=n_tests, n_steps=1)
        err_mat_syn_1[i_z,i_f,i_a,i_s,i_p] = err

        cap, _ = rnn.get_tolerance('gauss1masked', p*0.2, n_tests=n_tests, n_steps=50)
        cap_mat_syn_1mask[i_z,i_f,i_a,i_s,i_p] = cap
        _, err = rnn.get_tolerance('gauss1masked', p*0.2, n_tests=n_tests, n_steps=1)
        err_mat_syn_1mask[i_z,i_f,i_a,i_s,i_p] = err

    print(f'computed tol!')

# Save data
data = {}
data['date'] = rnn.get_date()
data['params'] = rnn.params
data['z'] = z_vec
data['f'] = f_vec
data['a'] = a_vec
data['p'] = p_vec
data['seed'] = seed_vec
data['min_margin_mean'] = min_margin_mean
data['min_margin_std'] = min_margin_std
data['avg_margin_mean'] = avg_margin_mean
data['avg_margin_std'] = avg_margin_std
data['rho_mean'] = rho_mean
data['rho_std'] = rho_std
data['L23_margin_mean'] = L23_margin_mean
data['L23_margin_std'] = L23_margin_std
data['L24_margin_mean'] = L24_margin_mean
data['L24_margin_std'] = L24_margin_std
data['syn_margin_mean'] = syn_margin_mean
data['syn_margin_std'] = syn_margin_std
data['l0_mean'] = l0_mean
data['l0_std'] = l0_std
data['l1_mean'] = l1_mean
data['l1_std'] = l1_std
data['l2_mean'] = l2_mean
data['l2_std'] = l2_std
data['sym'] = sym_mat
data['theta'] = theta_mat
data['bias'] = bias_mat
data['conv_time'] = conv_mat
data['w'] = w_mat
data['u'] = u_mat
data['cap_iso'] = cap_mat_iso
data['err_iso'] = err_mat_iso
data['cap_bal'] = cap_mat_bal
data['err_bal'] = err_mat_bal
data['cap_syn_z'] = cap_mat_syn_z
data['err_syn_z'] = err_mat_syn_z
data['cap_syn_1'] = cap_mat_syn_1
data['err_syn_1'] = err_mat_syn_1
data['cap_syn_1mask'] = cap_mat_syn_1mask
data['err_syn_1mask'] = err_mat_syn_1mask

torch.save(data, 'syn-con.pt')
