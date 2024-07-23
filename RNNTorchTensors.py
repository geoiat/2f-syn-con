#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-11-10

@author: Georgios Iatropoulos
"""

import math
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import erfcinv
from datetime import datetime


class RNN:

    def __init__(self, N, M, f, inh_ratio, z, u0, theta0, length, eta, eta_theta, seed,
                 n_epochs, norm_type, weight_type, optimizer_type, pattern_type='binomial', mem_type='attractor',
                 inh_type='unbalanced', with_bias=False, beta=100, kernel_degree=1, record_on=False):

        # Save parameters in dictionary
        self.params = locals()
        self.params.pop('self')
        
        # Set the parameters
        self.N = N
        self.M = M
        self.f = f
        self.inh_ratio = inh_ratio
        self.z = z
        self.u0 = u0
        self.theta0 = theta0
        self.length = length
        self.eta = eta
        self.eta_theta = eta_theta
        self.seed = seed
        self.n_epochs = n_epochs
        self.norm_type = norm_type
        self.weight_type = weight_type
        self.optimizer_type = optimizer_type
        self.pattern_type = pattern_type
        self.mem_type = mem_type
        self.inh_type = inh_type
        self.with_bias = with_bias
        self.beta = beta
        self.kernel_degree = kernel_degree
        self.record_on = record_on

        self.dynamic_inh = False
        
        #############################################################################################
        ###########      Variables about network size, stimulus etc       ###########################
        #############################################################################################

        # Population sizes
        self.N_inh = round(self.N*self.inh_ratio)
        self.N_exc = self.N - self.N_inh

        # Population activity
        if self.inh_type=='balanced':
            self.f_inh = self.f/(2.*self.inh_ratio)
            self.f_exc = self.f/(2.*(1.-self.inh_ratio))
        elif self.inh_type=='unbalanced':
            self.f_inh = self.f
            self.f_exc = self.f
        
        #############################################################################################
        ###########      Miscellaneous definitions     ##############################################
        #############################################################################################

        # Seed PyTorch/NumPy
        torch.manual_seed(seed=self.seed)
        self.rng = np.random.default_rng(self.seed)

        # Note which device that is available
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    
    def init_patterns(self):
        # Pattern matrix
        self.X = np.zeros((self.N,self.M))

        # Patterns where each neuron is Bernoulli with p=f
        if self.pattern_type=='bernoulli':
            X_exc = self.rng.binomial(n=1, p=self.f_exc, size=(self.N_exc,self.M))
            X_inh = -self.rng.binomial(n=1, p=self.f_inh, size=(self.N_inh,self.M))

            # no dead neurons
            for i in range(self.N_exc):
                while X_exc[i,:].sum()==0:
                    X_exc[i,:] = self.rng.binomial(n=1, p=self.f_exc, size=self.M)
            for i in range(self.N_inh):
                while X_inh[i,:].sum()==0:
                    X_inh[i,:] = self.rng.binomial(n=1, p=self.f_inh, size=self.M)
        
        # Patterns where activity is exactly f
        elif self.pattern_type=='exact':
            # Create matrices for exc and inh part of pattern, with enough 1s and 0s
            X_exc = np.zeros((self.N_exc,self.M))
            X_inh = np.zeros((self.N_inh,self.M))

            n_ones_exc = round(self.N_exc*self.f_exc)
            X_exc[:n_ones_exc,:] = 1.

            n_ones_inh = round(self.N_inh*self.f_inh)
            X_inh[:n_ones_inh,:] = -1.

            # Shuffle 1s and 0s
            for mu in range(self.M):
                self.rng.shuffle(X_exc[:,mu])
                self.rng.shuffle(X_inh[:,mu])
        
        # Merge matrices
        np.concatenate((X_exc, X_inh), axis=0, out=self.X)
        
        # Create all polynomial features
        kernel_transformer = PolynomialFeatures(self.kernel_degree, include_bias=False, interaction_only=True)
        self.X = kernel_transformer.fit_transform(self.X.T).T
        self.d, _ = self.X.shape

        # Label matrix
        self.Y = 2*self.X - 1

        # To GPU
        self.X = torch.from_numpy(self.X).to(dtype=torch.float, device=self.device)
        self.Y = torch.from_numpy(self.Y).to(dtype=torch.float, device=self.device)

        # For sequences
        if self.mem_type=='sequence':
            self.Y = self.Y.roll(-1, dims=1)
    

    def init_tensors(self):
        # Weight tensors
        if self.weight_type == 'hadamard':
            self.U0 = torch.zeros((self.N,self.d,self.z), device=self.device).fill_(self.u0)
            self.U = torch.zeros((self.N,self.d,self.z), device=self.device)
            #torch.normal(mean=self.U0, std=self.U0/2, out=self.U)
            self.U.uniform_(self.u0*0.7, self.u0*1.3)
            #self.U.uniform_(self.u0*0.5, self.u0*1.5)
            #self.U = torch.rand(size=self.U0.shape, device=self.device)*self.u0*2*0.3 + self.u0*(1-0.3)
            #self.U *= (torch.rand(size=self.U.shape, device=self.device) + 0.5).pow(1/self.z)
            self.diag_mask = 1 - torch.eye(self.N, device=self.device).reshape(self.N,self.d,1)
            self.grad_mask = torch.ones(self.N,self.d,self.z,self.z, device=self.device)
            self.grad_tmp = torch.ones(self.N,self.d,self.z,self.z, device=self.device)
            for i_z in range(self.z):
                self.grad_mask[:,:,i_z,i_z] = 0
            #self.U.normal_(mean=self.u0, std=self.u0/10)

        elif self.weight_type == 'vaskevicius':
            self.U = torch.zeros((self.N,self.d,2), device=self.device)
            self.U[:,:,0].fill_(2**(1/self.z)*self.u0)
            self.U[:,:,1].fill_(self.u0)
            self.Ones = torch.zeros((self.N,self.d,2), device=self.device)
            self.Ones[:,:,0].fill_(1.)
            self.Ones[:,:,1].fill_(-1.)

        elif self.weight_type == 'winnow':
            self.U0 = torch.zeros((self.N,self.d,1), device=self.device).fill_(self.u0)
            self.U = torch.zeros((self.N,self.d,1), device=self.device)
            torch.normal(mean=self.U0, std=self.U0/100, out=self.U)
            #self.U.normal_(mean=self.u0, std=self.u0/10)

        self.W = torch.zeros((self.N,self.d), device=self.device)
        self.dWdU = torch.zeros(self.U.shape, device=self.device)
        self.compute_W()

        # Auxiliary tensors
        self.one = torch.tensor(1., device=self.device)
        self.ones = torch.ones(self.W.shape, device=self.device)
        self.zero = torch.tensor(0., device=self.device)

        # Threshold vector
        self.theta = torch.zeros(self.N, device=self.device).fill_(self.theta0)
        self.d_theta = torch.tensor(0., device=self.device)

        # Weight mask tensors
        self.Mask_bool = torch.zeros(self.W.shape, dtype=torch.bool, device=self.device)
        self.Mask_big = torch.mul(self.Mask_bool, self.ones)
        self.Mask_small = torch.mul(self.Mask_bool, self.ones)

        # Distortion vector for testing robustness
        self.noise = torch.zeros((self.N,self.M), device=self.device)
        self.noise_U = torch.zeros(self.U.shape, device=self.device)


    def init_recorders(self):
        self.rec = {}
        
        #self.rec['loss'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['err'] = torch.zeros(self.n_epochs+1, device=self.device)
        #self.rec['acc'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['margin'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['rho'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['l0'] = torch.zeros(self.n_epochs+1, device=self.device)
        #self.rec['l0_exc'] = torch.zeros(self.n_epochs+1, device=self.device)
        #self.rec['l0_inh'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['l1'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['l2'] = torch.zeros(self.n_epochs+1, device=self.device)
        #self.rec['H'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['w'] = torch.zeros((self.N**2,100), device=self.device)
                
        self.rec['t'] = torch.arange(0, self.n_epochs+1, device=self.device)


    def record_epoch(self, epoch):
        self.rec['l0'][epoch] = self.get_l0()[0]
        #self.rec['l0_exc'][epoch] = self.get_l0(conn_type='exc')
        #self.rec['l0_inh'][epoch] = self.get_l0(conn_type='inh')
        self.rec['l1'][epoch] = self.get_l1()[0]
        self.rec['l2'][epoch] = self.get_l2()[0]
        #self.rec['H'][epoch] = self.H
        self.rec['margin'][epoch] = self.get_margin(self.X, self.y)[0]
        self.rec['rho'][epoch] = self.get_rho(self.X, self.y)[0]
        #self.rec['loss'][epoch] = self.get_loss(self.X, self.y)
        self.rec['err'][epoch] = self.get_error(self.X, self.y)
        #self.rec['acc'][epoch] = self.get_accuracy(self.X, self.y)

    
    def record_weights(self, epoch):
        self.rec['w'][:,epoch] = self.W.flatten()


    def forward(self, S, T):
        S_ = S * 1.
        H_ = S * 0.
        for t in range(T):
            torch.matmul(self.W, S_, out=H_)
            torch.sub(H_, self.theta[:,None], out=H_)
            torch.heaviside(H_, self.zero, out=S_)
            if self.dynamic_inh:
                self.d_theta = torch.sort(H_, dim=0)[0][int(self.N*(1-self.f))-1:int(self.N*(1-self.f)),:].mean(dim=0)
                torch.sub(H_, self.d_theta, out=H_)
                torch.heaviside(H_, self.zero, out=S_)
            else:
                self.d_theta *= 0.
        return S_


    def compute_W(self):
        # Collapse the U-tensor into the W-tensor
        if self.weight_type == 'hadamard':
            torch.prod(self.U, dim=2, out=self.W)

        elif self.weight_type == 'vaskevicius':
            torch.sum(self.Ones*self.U**(self.z), dim=2, out=self.W)

        elif self.weight_type == 'winnow':
            torch.pow(self.U[:,:,0], self.z, out=self.W)


    def compute_dWdU(self):
        if self.weight_type == 'hadamard':
            #self.dWdU = self.U[:,:,:,None].pow(self.grad_mask).prod(dim=2)
            torch.pow(self.U[:,:,:,None], self.grad_mask, out=self.grad_tmp)
            torch.prod(self.grad_tmp, dim=2, out=self.dWdU)
            #torch.prod(torch.pow(self.U[:,:,None], self.u_cube).T, dim=2, out=self.dWdU)
                
        elif self.weight_type == 'vaskevicius':
            torch.mul(self.Ones*self.z, self.U**(self.z-1), out=self.dWdU)
            
        elif self.weight_type == 'winnow':
            torch.pow(self.U, self.z-1, out=self.dWdU)
            #self.dWdU *= self.z#**(-self.z+1)


    def clip(self):
        # Keep the weights positive
        if self.weight_type == 'hadamard':
            self.U *= self.Mask_big[:,:,None]
            self.U.clamp_min_(1e-20)
            #torch.relu_(self.U)

            # Remove self-connections
            self.U *= self.diag_mask

        elif self.weight_type == 'vaskevicius':
            self.U.clamp_min_(1e-20)
            torch.minimum(self.U[:,:,0], self.U[:,:,1], out=self.U[:,:,1]) # u2 must be smaller than u1

        elif self.weight_type == 'winnow':
            self.U *= self.Mask_big[:,:,None]
            self.U.clamp_min_(1e-20)
            #torch.relu_(self.U)

            # Remove self-connections
            self.U[:,:,0].fill_diagonal_(0.)

        self.compute_W()


    def normalize(self):
        if self.norm_type == 'u':
            #U_factor =  torch.maximum(self.one, torch.linalg.norm(self.U.flatten(), ord=2)/self.length)
            U_factor = self.U.pow(2).sum(dim=(1,2), keepdim=True)/self.length/self.U.shape[2]
            #self.theta /= U_factor
            self.U /= U_factor.pow(0.5)#.pow(self.Mask_big[:,:,None])
        
        else:
            #w_factor =  torch.maximum(self.one, torch.linalg.norm(self.w, ord=self.norm_type)/self.length)
            W_factor = self.W.pow(self.norm_type).sum(dim=1, keepdim=True)/self.length
            #self.theta /= W_factor
            self.U /= W_factor[:,:,None].pow(1/(self.norm_type*self.z))#.pow(self.Mask_big[:,:,None])

        self.compute_W()


    def scale_weights(self, factor):
        self.U *= factor**(1/self.z)
        self.compute_W()


    def scale_weights_to_ref(self, ref):
        self.compute_masks()
        # normalize w.r.t. theta
        if ref == 'theta':
            self.U /= self.theta[:,None,None].pow(1/self.z)#.pow(self.Mask_big[:,:,None])
            self.compute_W()
            self.theta /= self.theta

        # normalize w.r.t. mean input
        elif ref == 'mean-i':
            self.theta /= 10*self.W.mean(dim=1)
            self.U /= (10*self.W.mean(dim=1)[:,None,None]).pow(1/self.z)#.pow(self.Mask_big[:,:,None])
            self.compute_W()

        # normalize w.r.t. mean w
        elif ref == 'mean-w':
            self.theta /= ( 10*self.W.sum(dim=1)/self.Mask_big.sum(dim=1) )
            self.U /= ( 10*self.W.sum(dim=1)/self.Mask_big.sum(dim=1) )[:,None,None].pow(1/self.z)
            #self.theta /= 10*self.W.mean(dim=1)
            #self.U /= (10*self.W.mean(dim=1)[:,None,None]).pow(1/self.z)#.pow(self.Mask_big[:,:,None])
            self.compute_W()

        # normalize w.r.t. mean u
        elif ref == 'mean-u':
            self.theta /= ( 10*self.U.sum(dim=(1,2))/(self.Mask_big.sum(dim=1)*self.z) ).pow(self.z)
            self.U /= ( 10*self.U.sum(dim=(1,2))/(self.Mask_big.sum(dim=1)*self.z) )[:,None,None]
            #self.theta /= (10*self.U.mean(dim=(1,2))).pow(self.z)
            #self.U /= (10*self.U.mean(dim=(1,2), keepdim=True))#.pow(self.Mask_big[:,:,None])
            self.compute_W()

    
    def compute_masks(self):
        torch.gt(self.W, 1e-10, out=self.Mask_bool)
        torch.mul(self.Mask_bool, self.ones, out=self.Mask_big)
        torch.sub(self.ones, self.Mask_big, out=self.Mask_small)


    def get_theta(self):
        return self.theta.mean()

    
    def get_bias(self):
        h_mean = self.f*self.W.sum(dim=1)
        h_var =  self.f*(1-self.f)*self.W.pow(2).sum(dim=1) # (self.W.std(dim=1)**2 + self.W.mean(dim=1)**2)
        #h_var = self.f*self.N*self.W.var(dim=1)
        #return h_mean + torch.sqrt(2*h_var)*erfcinv(2*self.f)
        return h_mean + torch.sqrt(2*h_var)*torch.special.erfinv(self.one - 2*self.f*self.one)

    
    def get_error(self, X=None, Y=None):
        if X is None: X = self.X
        if Y is None: Y = self.Y
        return ( (1 - Y*torch.sign(self.W@X - self.theta[:,None]))/2 ).mean()
    
    
    def get_accuracy(self, X=None, Y=None):
        if X is None: X = self.X
        if Y is None: Y = self.Y
        return ( (self.X-self.f)*(torch.heaviside(self.W@X - self.theta[:,None], self.zero)-self.f)/(self.N*self.f*(1-self.f)) ).sum(dim=0).mean()


    def get_capacity(self, X=None, Y=None):
        if X is None: X = self.X
        if Y is None: Y = self.Y
        err_vec = ( (1 - Y*torch.sign(self.W@X - self.theta[:,None]))/2 ).sum(dim=0)
        return torch.sum(err_vec <= 0.2*self.N*self.f)/self.M
    

    def get_margin(self, X=None, Y=None):
        return self.get_min_margin(X, Y)
    
    
    def get_min_margin(self, X=None, Y=None):
        if X is None: X = self.X
        if Y is None: Y = self.Y
        #return ( Y*(self.W@X - self.theta[:,None])/torch.linalg.norm(self.W, dim=1, keepdim=True) ).min(dim=1)[0].mean()
        margin_vec = ( Y*(self.W@X - self.theta[:,None])/torch.linalg.norm(self.W, dim=1, keepdim=True) ).min(dim=1)[0]
        return margin_vec.mean(), margin_vec.std()
    
    
    def get_avg_margin(self, X=None, Y=None):
        if X is None: X = self.X
        if Y is None: Y = self.Y
        #return ( Y*(self.W@X - self.theta[:,None])/torch.linalg.norm(self.W, dim=1, keepdim=True) ).mean(dim=1).mean()
        margin_vec = ( Y*(self.W@X - self.theta[:,None])/torch.linalg.norm(self.W, dim=1, keepdim=True) ).mean(dim=1)
        return margin_vec.mean(), margin_vec.std()


    def get_min_radius(self, X=None, Y=None):
        if X is None: X = self.X
        if Y is None: Y = self.Y
        #return ( Y*(self.W@X - self.theta[:,None])/torch.linalg.norm(self.W, dim=1, keepdim=True) ).min(dim=1)[0].mean()
        radius_vec = ( Y*(self.W@X - self.theta[:,None])/torch.linalg.norm(self.W, dim=1, keepdim=True) ).min(dim=0)[0]
        return radius_vec#.mean(), radius_vec.std()
    
    
    def get_avg_radius(self, X=None, Y=None):
        if X is None: X = self.X
        if Y is None: Y = self.Y
        #return ( Y*(self.W@X - self.theta[:,None])/torch.linalg.norm(self.W, dim=1, keepdim=True) ).mean(dim=1).mean()
        radius_vec = ( Y*(self.W@X - self.theta[:,None])/torch.linalg.norm(self.W, dim=1, keepdim=True) ).mean(dim=0)
        return radius_vec#.mean(), radius_vec.std()
        
    
    def get_rho(self, X=None, Y=None):
        if X is None: X = self.X
        if Y is None: Y = self.Y
        rho_vec = ( Y*(self.W@X - self.theta[:,None]) / (self.W.mean(dim=1, keepdim=True)*math.sqrt(self.N*self.f*(1-self.f))) ).min(dim=1)[0]
        #rho_vec = ( 1 / ( self.W.mean(dim=1, keepdim=True)*math.sqrt(self.N*self.f*(1-self.f))/(Y*(self.W@X - self.theta[:,None])) ) ).min(dim=1)[0]
        return rho_vec.mean(), rho_vec.std()


    def get_norm_margin(self, norm, X=None, Y=None):
        if X is None: X = self.X
        if Y is None: Y = self.Y
        #return ( Y*(self.W@X - self.theta[:,None])/torch.linalg.norm(self.W, dim=1, keepdim=True) ).min(dim=1)[0].mean()
        margin_vec = ( Y*(self.W@X - self.theta[:,None])/torch.linalg.norm(self.W, ord=norm, dim=1, keepdim=True) ).min(dim=1)[0]
        return margin_vec.mean(), margin_vec.std()


    def get_syn_margin(self, X=None, Y=None):
        if X is None: X = self.X
        if Y is None: Y = self.Y
        norm = 2 - 2/self.z
        #return ( Y*(self.W@X - self.theta[:,None])/torch.linalg.norm(self.W, dim=1, keepdim=True) ).min(dim=1)[0].mean()
        if norm==0:
            margin_vec = ( Y*(self.W@X - self.theta[:,None])/np.sqrt(self.N) ).min(dim=1)[0]
        else:
            margin_vec = ( Y*(self.W@X - self.theta[:,None])/self.W.pow(norm).sum(dim=1, keepdim=True).sqrt() ).min(dim=1)[0]
        return margin_vec.mean(), margin_vec.std()

    
    def get_tolerance(self, noise_type, noise_level, n_tests, n_steps, X=None, Y=None):
        if X is None: X = self.X
        if Y is None: Y = self.Y
        cap = 0. # capacity
        err = 0. # error rate
        if noise_type == 'bernoulli':
            p = noise_level
            p_mat = self.noise*0 + p
            for _ in range(n_tests):
                self.noise.bernoulli_(p=1-p_mat).sub_(0.5).mul_(2)
                X_noise = (X - 0.5)*self.noise + 0.5
                S_end = self.forward(S=X_noise, T=n_steps-1)
                cap += self.get_capacity(S_end, Y)/n_tests
                err += self.get_error(S_end, Y)/n_tests

        elif noise_type == 'bernoulli-balanced':
            p = noise_level
            p_mat = torch.where(X>0, p/self.f, p/(1-self.f))
            for _ in range(n_tests):
                self.noise.bernoulli_(p=1-p_mat).sub_(0.5).mul_(2)
                X_noise = (X - 0.5)*self.noise + 0.5
                S_end = self.forward(S=X_noise, T=n_steps-1)
                cap += self.get_capacity(S_end, Y)/n_tests
                err += self.get_error(S_end, Y)/n_tests

        elif noise_type == 'bits':
            n = int(noise_level*self.N)
            for _ in range(n_tests):
                self.noise.fill_(1)
                for mu in range(self.M):
                    xi = X[:,mu]
                    idx_1 = xi.nonzero().flatten()
                    idx_0 = (xi-1).nonzero().flatten()
                    idx_flip_10 = idx_1[torch.randperm(idx_1.shape[0], device=self.device)[:n]]
                    idx_flip_01 = idx_0[torch.randperm(idx_0.shape[0], device=self.device)[:n]]
                    self.noise[idx_flip_01,mu] *= -1
                    self.noise[idx_flip_10,mu] *= -1
                    X_noise = (X - 0.5)*self.noise + 0.5

                S_end = self.forward(S=X_noise, T=n_steps-1)
                cap += self.get_capacity(S_end, Y)/n_tests
                err += self.get_error(S_end, Y)/n_tests
        
        elif noise_type == 'gauss':
            std = noise_level
            U_start = self.U.clone()
            self.compute_masks()
            if self.z==1:
                noise_mask = torch.heaviside(self.W - 1e-10, self.zero)
            else:
                noise_mask = torch.heaviside(self.W - 1e-10, self.zero)
            for _ in range(n_tests):
                self.noise_U.normal_(mean=0, std=std)
                #self.noise_U.uniform_(-std, std)
                self.U = U_start + self.noise_U#*self.Mask_big[:,:,None]
                #self.U[:,:,0] = U_start[:,:,0] + self.noise_U[:,:,0]*self.Mask_big
                #self.clip()
                self.U.clamp_min_(1e-20)
                self.compute_W()
                S_end = self.forward(S=X, T=n_steps-1)
                cap += self.get_capacity(S_end, Y)/n_tests
                err += self.get_error(S_end, Y)/n_tests

            self.U = U_start.clone()
            self.compute_W()

        elif noise_type == 'gauss1':
            std = noise_level
            U_start = self.U.clone()
            self.compute_masks()
            #self.compute_dWdU()
            #if self.z==1:
            #    noise_mask = torch.heaviside(self.W - 1e-10, self.zero)
            #else:
            #    noise_mask = torch.heaviside(self.W - 1e-10, self.zero)
            for _ in range(n_tests):
                self.noise_U.normal_(mean=0, std=std)
                #self.noise_U.uniform_(-std, std)
                #self.U = U_start + self.noise_U*self.dWdU
                self.U[:,:,0] = U_start[:,:,0] + self.noise_U[:,:,0]*self.Mask_big + self.noise_U[:,:,0].abs()*self.Mask_small
                #self.clip()
                self.U.clamp_min_(1e-20)
                self.compute_W()
                S_end = self.forward(S=X, T=n_steps-1)
                cap += self.get_capacity(S_end, Y)/n_tests
                err += self.get_error(S_end, Y)/n_tests

            self.U = U_start.clone()
            self.compute_W()

        elif noise_type == 'gauss1masked':
            std = noise_level
            U_start = self.U.clone()
            self.compute_masks()
            #self.compute_dWdU()
            #if self.z==1:
            #    noise_mask = torch.heaviside(self.W - 1e-10, self.zero)
            #else:
            #    noise_mask = torch.heaviside(self.W - 1e-10, self.zero)
            for _ in range(n_tests):
                self.noise_U.normal_(mean=0, std=std)
                #self.noise_U.uniform_(-std, std)
                #self.U = U_start + self.noise_U*self.dWdU
                self.U[:,:,0] = U_start[:,:,0] + self.noise_U[:,:,0]*self.Mask_big
                #self.clip()
                self.U.clamp_min_(1e-20)
                self.compute_W()
                S_end = self.forward(S=X, T=n_steps-1)
                cap += self.get_capacity(S_end, Y)/n_tests
                err += self.get_error(S_end, Y)/n_tests

            self.U = U_start.clone()
            self.compute_W()
        
        return cap, err
    
    
    def get_balance_ratio(self):
        return (self.N*self.f*self.W.mean(dim=1)/self.theta).mean()


    def get_singular_value(self):
        return torch.linalg.norm(self.W, ord=2)


    def get_l2(self):
        l2_vec = torch.linalg.norm(self.W, ord=2, dim=1)
        return l2_vec.mean(), l2_vec.std()
    
    
    def get_l1(self):
        l1_vec = torch.linalg.norm(self.W, ord=1, dim=1)
        return l1_vec.mean(), l1_vec.std()
    
    
    def get_l0(self, conn_type=None):
        threshold = 1e-10 # self.u0**(self.z) * 1e-2
        if conn_type==None:
            l0_vec = torch.sum(torch.gt(self.W, threshold), dim=1)/self.N
            return l0_vec.mean(), l0_vec.std()
        elif conn_type=='l2':
            l0_vec = torch.sum(torch.gt(self.W, self.eta), dim=1)/self.N
            return l0_vec.mean(), l0_vec.std()
        elif conn_type=='exc':
            return torch.sum(torch.gt(self.W[:self.N_exc], threshold))/self.N_exc
        elif conn_type=='inh' and self.inh_ratio>0:
            return torch.sum(torch.gt(self.W[self.N_exc:self.N], threshold))/self.N_inh
        elif conn_type=='inh' and self.inh_ratio==0:
            return float('nan')
    
    
    def get_I(self, population=None):
        if self.kernel_degree == 1:
            if population==None:
                return -self.N*self.M*(self.f*torch.log2(self.f) + (1-self.f)*torch.log2(1-self.f))
            elif population=='exc':
                return -self.N_exc*self.M*(self.f_exc*torch.log2(self.f_exc) + (1-self.f_exc)*torch.log2(1-self.f_exc))
            elif population=='inh' and self.f_inh>0:
                return -self.N_inh*self.M*(self.f_inh*torch.log2(self.f_inh) + (1-self.f_inh)*torch.log2(1-self.f_inh))
            elif population=='inh' and self.f_inh==0:
                return 0
        else:
            return -self.N*self.M*(self.f*torch.log2(self.f) + (1-self.f)*torch.log2(1-self.f)) + \
                   -(self.d - self.N)*self.M*(self.f**2*torch.log2(self.f**2) + (1-self.f**2)*torch.log2(1-self.f**2))


    def get_symmetry(self):
        W_clip = self.W*self.Mask_big
        S = (W_clip - W_clip.T).abs()/(W_clip + W_clip.T)
        return 1 - np.nanmean( S.cpu().numpy() )


    def get_sparseness(self, variant='lifetime'):
        R_mat = self.W@self.X #- self.theta[:,None]
        
        # no dead neurons
        #idx_alive = self.X.sum(dim=1) > 0
        #R_mat = R_mat[idx_alive,:]

        # normalize
        #R_mat = R_mat - (R_mat.min(dim=1, keepdim=True)[0])
        #R_mat = R_mat / (R_mat.max(dim=1, keepdim=True)[0])

        if variant=='population':
            R_mat = R_mat.T
        
        n_samples = R_mat.shape[1]
        A = (R_mat.mean(dim=1).pow(2)) / (R_mat.pow(2).mean(dim=1))
        sp_vec = (1 - A)/(1 - 1/n_samples)
        return sp_vec.mean(), sp_vec.std()


    def get_u_corr(self, dim):
        U_ = self.U - self.U.mean(dim=0, keepdim=True)
        U_ /= torch.linalg.norm(U_, dim=0, keepdim=True)
        if dim == 1:
            r = (torch.einsum('ikj,ilj->j', U_, U_) - self.N)/(self.N*(self.N-1))
        elif dim == 2:
            r = (torch.einsum('ijk,ijl->j', U_, U_) - self.z)/(self.z*(self.z-1))
        return r.mean()


    def get_date(self):
        return datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            
    
    def print_progress(self, epoch, interval):
        if epoch % interval == 0:
            print(f'Epoch: {epoch:7}/{self.n_epochs} || '+
                  f'theta: {self.get_theta().item():7.3f} || '+
                  f'bias: {self.get_bias().mean().item():7.3f} || '+
                  f'l0: {self.get_l0()[0].item():7.5f} || '+
                  f'l1: {self.get_l1()[0].item():7.3f} || '+
                  f'l2: {self.get_l2()[0].item():7.3f} || '+
                  f'norm: {self.W.pow(2/self.z).sum(dim=1).mean().item():7.3f} || '+
                  f'rho: {self.get_rho()[0].item():7.3f} || '+
                  f'margin: {self.get_margin()[0].item():8.5f} || '+
                  f'err: {self.get_error().item():7.3}')


    def save(self, path):
        torch.save(self.rec, path)
        

    def optimize(self):
        if self.optimizer_type == 'bp':
            H_ = torch.zeros((self.N,self.M), device=self.device)
            Mask_mu = torch.zeros((self.N,self.M), device=self.device)
            N_vec = torch.arange(self.N, device=self.device)
            self.compute_masks()
            self.normalize()
            self.theta = self.get_bias()
            self.conv_time = -1
            self.print_progress(0, interval=1e4)

            if self.record_on:
                self.record_epoch(0)

            #H_means = (self.W@self.X - self.theta[:,None]).abs().mean(dim=1, keepdim=True)

            margin_old = self.get_margin()[0]
            l0_old = self.get_l0()[0]
            for epoch in range(1, self.n_epochs+1):
                if epoch%1e4==0:
                    margin_new = self.get_margin()[0]
                    l0_new = self.get_l0()[0]
                    norm_new = self.W.pow(2/self.z).sum(dim=1).mean()
                    if (l0_old-l0_new).abs()<0.00010 and (margin_old-margin_new).abs()<0.00010 and l0_new<0.99 and self.get_error()==0 and self.conv_time<0 and (norm_new-self.length).abs()<0.02:
                        self.conv_time = epoch
                        break
                    else:
                        l0_old = l0_new
                        margin_old = margin_new

                self.print_progress(epoch, interval=1e4)
                
                if self.get_error()>0:
                    # Find weakest weights
                    torch.matmul(self.W, self.X, out=H_)
                    torch.sub(H_, self.theta[:,None], out=H_)
                    torch.mul(H_, self.Y, out=H_)
                    mu_min = torch.argmin(H_, dim=1)
                    Mask_mu.fill_(0.)
                    Mask_mu[N_vec, mu_min] += 1.
                    X_min = torch.matmul(Mask_mu, self.X.T)
                    y_min = self.Y[N_vec, mu_min]
                    H_means = H_.abs().mean(dim=1, keepdim=True)

                    # Update weights
                    self.compute_dWdU()
                    self.compute_masks() #torch.heaviside(self.W - 1e-10, self.zero, out=self.Mask_big)
                    self.U += self.eta*y_min[:,None,None]*X_min[:,:,None]*self.dWdU*self.Mask_big[:,:,None] #+ 10*self.eta*(1 - self.U.pow(2).sum(dim=1, keepdim=True)/self.length)*self.U
                    self.clip()
                    self.normalize()

                else:
                    torch.matmul(self.W, self.X, out=H_)
                    torch.sub(H_, self.theta[:,None], out=H_)
                    Y_self = torch.sign(H_)
                    #torch.mul(H_, Y_self, out=H_)
                    #Loss = torch.exp(-self.beta*H_.abs()/H_.abs().mean(dim=1, keepdim=True))
                    Loss = torch.exp(-self.beta*H_.abs()/H_means)
                    H_means = H_.abs().mean(dim=1, keepdim=True) #(self.W@X_sleep - self.theta[:,None]).abs().mean(dim=1, keepdim=True)
                    Mask_mu = (Loss*Y_self) / Loss.sum(dim=1, keepdim=True)
                    dLdW = torch.matmul(Mask_mu, self.X.T)
                    y_min = Mask_mu.sum(dim=1)

                    # Update weights
                    self.compute_dWdU()
                    self.compute_masks() #torch.heaviside(self.W - 1e-10, self.zero, out=self.Mask_big)
                    self.U += self.eta*dLdW[:,:,None]*self.dWdU*self.Mask_big[:,:,None] #+ 10*self.eta*(1 - self.U.pow(2).sum(dim=1, keepdim=True)/self.length)*self.U
                    self.clip()
                    self.normalize()

                # if epoch > self.n_epochs-100:
                #     self.record_weights(epoch - self.n_epochs - 1)

                # Update threshold
                if self.with_bias:
                    if True: #self.get_error().item()>0: # this is used only until we reach err=0
                        self.eta_theta_ = self.eta_theta#*np.exp(-10*epoch/self.n_epochs) # for quenching eta_theta
                        self.theta -= self.eta_theta_*y_min
                    else:
                        torch.matmul(self.W, self.X, out=H_)
                        H_ = torch.sort(H_, dim=1)[0]
                        idx_low = (H_-self.theta[:,None]).sign().diff(dim=1).argmax(dim=1)
                        idx_high = idx_low + 1
                        th_low = H_[N_vec, idx_low]
                        th_high = H_[N_vec, idx_high]
                        self.theta = (th_low + th_high)/2
                        self.theta += th_low.uniform_(-1e-6, 1e-6)
                    #self.theta -= self.eta_theta_*torch.sign(self.theta - self.get_bias())
                    #self.theta = self.get_bias()

                if self.record_on:
                    self.record_epoch(epoch)


        elif self.optimizer_type == 'bp-norm-off':
            H_ = torch.zeros((self.N,self.M), device=self.device)
            Mask_mu = torch.zeros((self.N,self.M), device=self.device)
            N_vec = torch.arange(self.N, device=self.device)
            self.compute_masks()
            self.normalize()
            self.theta = self.get_bias()
            self.conv_time = -1
            self.print_progress(0, interval=1e4)

            if self.record_on:
                self.record_epoch(0)

            margin_old = self.get_margin()[0]
            l0_old = self.get_l0()[0]
            for epoch in range(1, self.n_epochs+1):
                if epoch%1e4==0:
                    margin_new = self.get_margin()[0]
                    l0_new = self.get_l0()[0]
                    if (l0_old-l0_new).abs()<0.00010 and (margin_old-margin_new).abs()<0.00010 and l0_new<0.99 and self.get_error()==0 and self.conv_time<0:
                        self.conv_time = epoch
                        break
                    else:
                        l0_old = l0_new
                        margin_old = margin_new

                self.print_progress(epoch, interval=1e4)

                # Find weakest weights
                torch.matmul(self.W, self.X, out=H_)
                torch.sub(H_, self.theta[:,None], out=H_)
                torch.mul(H_, self.Y, out=H_)
                mu_min = torch.argmin(H_, dim=1)
                Mask_mu.fill_(0.)
                Mask_mu[N_vec, mu_min] += 1.
                X_min = torch.matmul(Mask_mu, self.X.T)
                y_min = self.Y[N_vec, mu_min]

                # Update weights
                self.compute_dWdU()
                self.compute_masks()
                self.U += self.eta*y_min[:,None,None]*X_min[:,:,None]*self.dWdU*self.Mask_big[:,:,None] #+ 10*self.eta*(1 - self.U.pow(2).sum(dim=1, keepdim=True)/self.length)*self.U
                self.clip()
                #self.normalize()

                if self.with_bias:
                    self.eta_theta_ = self.eta_theta#*np.exp(-10*epoch/self.n_epochs) # for quenching eta_theta
                    self.theta -= self.eta_theta_*y_min
                    #self.theta -= self.eta_theta_*torch.sign(self.theta - self.get_bias())
                    #self.theta = self.get_bias()

                if self.record_on:
                    self.record_epoch(epoch)


        elif self.optimizer_type == 'margin-perceptron':
            H_ = torch.zeros((self.N,self.M), device=self.device)
            Mask_mu = torch.zeros((self.N,self.M), device=self.device)
            N_vec = torch.arange(self.N, device=self.device)
            self.compute_masks()
            self.normalize()
            self.theta = self.get_bias()
            self.conv_time = -1

            if self.record_on:
                self.record_epoch(0)

            for epoch in range(1, self.n_epochs+1):
                K1 = ( self.Y*(self.W@self.X - self.theta[:,None]) ).min(dim=1)[0]/self.W.sum(dim=1)
                K1_mask = (K1*math.sqrt(self.N) < 0.25).float()

                if K1_mask.mean() <= 0.01:
                    self.conv_time = epoch
                    break

                self.print_progress(epoch, interval=1e3)

                # Find weakest weights
                torch.matmul(self.W, self.X, out=H_)
                torch.sub(H_, self.theta[:,None], out=H_)
                torch.mul(H_, self.Y, out=H_)
                mu_min = torch.argmin(H_, dim=1)
                Mask_mu.fill_(0.)
                Mask_mu[N_vec, mu_min] += 1.
                X_min = torch.matmul(Mask_mu, self.X.T)
                y_min = self.Y[N_vec, mu_min]

                # Update weights
                self.compute_dWdU()
                self.compute_masks()
                self.U += self.eta*K1_mask[:,None,None]*y_min[:,None,None]*X_min[:,:,None]*self.dWdU*self.Mask_big[:,:,None] #+ 10*self.eta*(1 - self.U.pow(2).sum(dim=1, keepdim=True)/self.length)*self.U
                self.clip()
                
                #self.normalize()
                #self.U += (self.length - self.W.sum(dim=1, keepdim=True))/self.N
                #self.clip()

                # if epoch > self.n_epochs-100:
                #     self.record_weights(epoch - self.n_epochs - 1)

                # Update threshold
                if self.with_bias:
                    if True: #self.get_error().item()>0: # this is used only until we reach err=0
                        self.eta_theta_ = self.eta_theta#*np.exp(-10*epoch/self.n_epochs) # for quenching eta_theta
                        self.theta -= self.eta_theta_*K1_mask*y_min
                    else:
                        torch.matmul(self.W, self.X, out=H_)
                        H_ = torch.sort(H_, dim=1)[0]
                        idx_low = (H_-self.theta[:,None]).sign().diff(dim=1).argmax(dim=1)
                        idx_high = idx_low + 1
                        th_low = H_[N_vec, idx_low]
                        th_high = H_[N_vec, idx_high]
                        self.theta = (th_low + th_high)/2
                        self.theta += th_low.uniform_(-1e-6, 1e-6)
                    #self.theta -= self.eta_theta_*torch.sign(self.theta - self.get_bias())
                    #self.theta = self.get_bias()

                if self.record_on:
                    self.record_epoch(epoch)



        
class ContinualRNN(RNN):

    def __init__(self, N, f, inh_ratio, z, u0, theta0, length,
                 eta_wake, eta_sleep, eta_theta, seed,
                 M_day, n_days, n_epochs_wake, n_epochs_sleep,
                 norm_type, weight_type, optimizer_type, pattern_type='bernoulli', mem_type='attractor',
                 inh_type='unbalanced', with_bias=False, exact_mode=True, beta=1, kernel_degree=1, record_on=False):

        # Save parameters in dictionary
        self.params = locals()
        self.params.pop('self')
        
        # Set the parameters
        self.N = N
        self.f = f
        self.inh_ratio = inh_ratio
        self.z = z
        self.u0 = u0
        self.theta0 = theta0
        self.length = length
        self.eta_wake = eta_wake
        self.eta_sleep = eta_sleep
        self.eta_theta = eta_theta
        self.seed = seed
        self.M_day = M_day
        self.n_days = n_days
        self.n_epochs_wake = n_epochs_wake
        self.n_epochs_sleep = n_epochs_sleep
        self.norm_type = norm_type
        self.weight_type = weight_type
        self.optimizer_type = optimizer_type
        self.pattern_type = pattern_type
        self.mem_type = mem_type
        self.inh_type = inh_type
        self.with_bias = with_bias
        self.exact_mode = exact_mode
        self.beta = beta
        self.kernel_degree = kernel_degree
        self.record_on = record_on

        self.dynamic_inh = True
        
        #############################################################################################
        ###########      Variables about network size, stimulus etc       ###########################
        #############################################################################################

        # Population sizes
        self.N_inh = round(self.N*self.inh_ratio)
        self.N_exc = self.N - self.N_inh

        # Population activity
        if self.inh_type=='balanced':
            self.f_inh = self.f/(2.*self.inh_ratio)
            self.f_exc = self.f/(2.*(1.-self.inh_ratio))
        elif self.inh_type=='unbalanced':
            self.f_inh = self.f
            self.f_exc = self.f

        # Pattern parameters
        self.M = self.M_day*self.n_days
        self.n_epochs = self.n_days*(self.M*self.n_epochs_wake+self.n_epochs_sleep)
        
        #############################################################################################
        ###########      Miscellaneous definitions     ##############################################
        #############################################################################################

        # Seed PyTorch/NumPy
        torch.manual_seed(seed=self.seed)
        self.rng = np.random.default_rng(self.seed)

        # Note which device that is available
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    
    def init_patterns(self):
        # List with daily patterns
        self.X_lst = []
        self.Y_lst = []
        kernel_transformer = PolynomialFeatures(self.kernel_degree, include_bias=False, interaction_only=True)

        for i_day in range(self.n_days):
            # Pattern matrix
            X = np.zeros((self.N,self.M_day))

            # Patterns where each neuron is Bernoulli with p=f
            if self.pattern_type=='bernoulli':
                X_exc = self.rng.binomial(n=1, p=self.f_exc, size=(self.N_exc,self.M_day))
                X_inh = -self.rng.binomial(n=1, p=self.f_inh, size=(self.N_inh,self.M_day))
            
            # Patterns where activity is exactly f
            elif self.pattern_type=='exact':
                # Create matrices for exc and inh part of pattern, with enough 1s and 0s
                X_exc = np.zeros((self.N_exc,self.M_day))
                X_inh = np.zeros((self.N_inh,self.M_day))

                n_ones_exc = round(self.N_exc*self.f_exc)
                X_exc[:n_ones_exc,:] = 1.

                n_ones_inh = round(self.N_inh*self.f_inh)
                X_inh[:n_ones_inh,:] = -1.

                # Shuffle 1s and 0s
                for mu in range(self.M_day):
                    self.rng.shuffle(X_exc[:,mu])
                    self.rng.shuffle(X_inh[:,mu])
            
            # Merge matrices
            np.concatenate((X_exc, X_inh), axis=0, out=X)

            # Create all polynomial features
            X = kernel_transformer.fit_transform(X.T).T
            self.d, _ = X.shape

            # Label matrix
            Y = 2*X - 1
            
            # To GPU
            X = torch.from_numpy(X).to(dtype=torch.float, device=self.device)
            Y = torch.from_numpy(Y).to(dtype=torch.float, device=self.device)

            # For sequences
            if self.mem_type=='sequence':
                Y = Y.roll(-1, dims=1)

            self.X_lst.append(X)
            self.Y_lst.append(Y)
        
        # All patterns in one matrix
        self.X = torch.cat(self.X_lst, dim=1)
        self.Y = torch.cat(self.Y_lst, dim=1)

    
    def init_tensors(self):
        # Weight tensors
        if self.weight_type == 'hadamard':
            self.U0 = torch.zeros((self.N,self.d,self.z), device=self.device).fill_(self.u0)
            self.U = torch.zeros((self.N,self.d,self.z), device=self.device).fill_(self.u0)
            #torch.normal(mean=self.U0, std=self.U0/100, out=self.U)
            #self.U.normal_(mean=self.u0, std=self.u0/10)
            #self.U.uniform_(self.u0*0.98, self.u0*1.02)
            self.diag_mask = 1 - torch.eye(self.N, device=self.device).reshape(self.N,self.d,1)
            self.grad_mask = torch.ones(self.N,self.d,self.z,self.z, device=self.device)
            self.grad_tmp = torch.ones(self.N,self.d,self.z,self.z, device=self.device)
            for i_z in range(self.z):
                self.grad_mask[:,:,i_z,i_z] = 0

        elif self.weight_type == 'vaskevicius':
            self.U = torch.zeros((self.N,self.d,2), device=self.device)
            self.U[:,:,0].fill_(2**(1/self.z)*self.u0)
            self.U[:,:,1].fill_(self.u0)
            self.Ones = torch.zeros((self.N,self.d,2), device=self.device)
            self.Ones[:,:,0].fill_(1.)
            self.Ones[:,:,1].fill_(-1.)

        elif self.weight_type == 'winnow':
            self.U0 = torch.zeros((self.N,self.d,1), device=self.device).fill_(self.u0)
            self.U = torch.zeros((self.N,self.d,1), device=self.device).fill_(self.u0)
            #torch.normal(mean=self.U0, std=self.U0/100, out=self.U)
            #self.U.normal_(mean=self.u0, std=self.u0/10)
            #self.U0.uniform_(self.u0*0.7, self.u0*1.3)
            #self.U.uniform_(self.u0*0.7, self.u0*1.3)

        self.W = torch.zeros((self.N,self.d), device=self.device)
        self.dWdU = torch.zeros(self.U.shape, device=self.device)
        self.compute_W()

        # Auxiliary tensors
        self.one = torch.tensor(1., device=self.device)
        self.ones = torch.ones(self.W.shape, device=self.device)
        self.zero = torch.tensor(0., device=self.device)

        # Threshold vector
        self.theta = torch.zeros(self.N, device=self.device).fill_(self.theta0)
        self.d_theta = torch.tensor(0., device=self.device)

        # Weight mask tensors
        self.Mask_bool = torch.zeros(self.W.shape, dtype=torch.bool, device=self.device)
        self.Mask_big = torch.mul(self.Mask_bool, self.ones)
        self.Mask_small = torch.mul(self.Mask_bool, self.ones)

        # Distortion vector for testing robustness
        self.noise = torch.zeros((self.N,self.M), device=self.device)
        self.noise_U = torch.zeros(self.U.shape, device=self.device)

        # Surviving and dying weights (only for seed=700)
        self.idx_w_on = torch.tensor([56,  59, 127, 138, 139, 154, 159, 160, 172, 181], device=self.device)
        self.idx_w_off = torch.tensor([20, 21, 22, 23, 24, 25, 26, 27, 28, 29], device=self.device)


    def init_recorders(self):
        self.rec = {}
        
        #self.rec['loss'] = torch.zeros((self.n_epochs+1, self.n_days), device=self.device)
        self.rec['err'] = torch.zeros((self.n_epochs+1, self.n_days), device=self.device)
        self.rec['acc'] = torch.zeros((self.n_epochs+1, self.n_days), device=self.device)
        #self.rec['capacity'] = torch.zeros((self.n_epochs+1, self.n_days), device=self.device)
        self.rec['RR'] = torch.zeros((self.n_epochs+1, self.n_days), device=self.device)
        self.rec['margin_mean'] = torch.zeros((self.n_epochs+1, self.n_days), device=self.device)
        self.rec['margin_std'] = torch.zeros((self.n_epochs+1, self.n_days), device=self.device)
        self.rec['rho_mean'] = torch.zeros((self.n_epochs+1, self.n_days), device=self.device)
        self.rec['rho_std'] = torch.zeros((self.n_epochs+1, self.n_days), device=self.device)
        self.rec['l0_mean'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['l0_std'] = torch.zeros(self.n_epochs+1, device=self.device)
        #self.rec['l0_exc'] = torch.zeros(self.n_epochs+1, device=self.device)
        #self.rec['l0_inh'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['l1'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['l2'] = torch.zeros(self.n_epochs+1, device=self.device)
        #self.rec['H'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['theta_mean'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['theta_all'] = torch.zeros((self.N,2), device=self.device)
        self.rec['theta'] = torch.zeros((30,self.n_epochs+1), device=self.device)
        self.rec['ymin'] = torch.zeros((30,self.n_epochs+1), device=self.device)
        self.rec['bias'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['d_theta'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['ucorr1'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['ucorr2'] = torch.zeros(self.n_epochs+1, device=self.device)
        
        self.rec['w'] = torch.zeros((self.N**2,31), device=self.device)
        self.rec['w_on'] = torch.zeros((10,self.n_epochs+1), device=self.device)
        self.rec['w_off'] = torch.zeros((10,self.n_epochs+1), device=self.device)
        self.rec['ltp-ratio_pop'] = torch.zeros(self.n_epochs+1, device=self.device)
        self.rec['ltp-ratio_time'] = torch.zeros(self.N, device=self.device)
        self.rec['replays'] = torch.zeros((self.M,2), device=self.device)
        self.rec['radius_min'] = torch.zeros((self.M,501), device=self.device)
        self.rec['radius_avg'] = torch.zeros((self.M,501), device=self.device)
        self.rec['cap_bal'] = torch.zeros((200,4), device=self.device)
        self.rec['cap_syn1'] = torch.zeros((200,4), device=self.device)
        self.rec['cap_synz'] = torch.zeros((200,4), device=self.device)
        self.p_vec = torch.linspace(0, 50-1, 200, device=self.device)/self.N
                
        self.rec['t'] = torch.zeros(self.n_epochs+1, device=self.device)


    def record_epoch(self, epoch):
        self.rec['l0_mean'][epoch], self.rec['l0_std'][epoch] = self.get_l0()
        #self.rec['l0_exc'][epoch] = self.get_l0(conn_type='exc')
        #self.rec['l0_inh'][epoch] = self.get_l0(conn_type='inh')
        self.rec['l1'][epoch] = self.get_l1()[0]
        self.rec['l2'][epoch] = self.get_l2()[0]
        self.rec['theta_mean'][epoch] = self.get_theta()
        self.rec['theta'][:,epoch] = self.theta[-30:].clone()
        self.rec['bias'][epoch] = self.get_bias().mean()
        self.rec['d_theta'][epoch] = self.d_theta.mean()
        self.rec['w_on'][:,epoch] = self.W[0,self.idx_w_on]
        self.rec['w_off'][:,epoch] = self.W[0,self.idx_w_off]
        self.rec['ucorr1'][epoch] = self.get_u_corr(dim=1)
        self.rec['ucorr2'][epoch] = self.get_u_corr(dim=2)
        #self.rec['H'][epoch] = self.H
        for day in range(self.n_days):
            #X_day = self.X_lst[day]
            #Y_day = self.Y_lst[day]
            self.rec['margin_mean'][epoch,day], self.rec['margin_std'][epoch,day] = self.get_margin()#X_day, Y_day)
            self.rec['rho_mean'][epoch,day], self.rec['rho_std'][epoch,day] = self.get_rho()#X_day, Y_day)
            #self.rec['loss'][epoch,day] = self.get_loss(X_day, Y_day)
            self.rec['err'][epoch,day] = self.get_error()#X_day, Y_day)
            self.rec['acc'][epoch,day] = self.get_accuracy()#X_day, Y_day)
            self.rec['RR'][epoch,day] = self.get_capacity()#X_day, Y_day)


    def forward_(self, S, T):
        S_ = S * 1.
        H_ = S * 0.
        for t in range(T):
            torch.matmul(self.W, S_, out=H_)
            torch.sub(H_, self.theta[:,None], out=H_)
            torch.heaviside(H_, self.zero, out=S_)
            if self.dynamic_inh and not S_.mean().isclose(self.one*self.f):
                self.d_theta = torch.sort(H_, dim=0)[0][int(self.N*(1-self.f))-1:int(self.N*(1-self.f)),:].mean(dim=0)
                torch.sub(H_, self.d_theta, out=H_)
                torch.heaviside(H_, self.zero, out=S_)
            else:
                self.d_theta *= 0.
        return S_


    def rec_tol(self, idx):
        self.dynamic_inh = False
        n_p = self.p_vec.shape[0]
        U_backup = self.U.clone()
        theta_backup = self.theta.clone()
        self.scale_weights_to_ref('mean-u')
        for i_p in range(n_p):
            p = self.p_vec[i_p].item()
            cap, _ = self.get_tolerance('bernoulli-balanced', p, n_tests=20, n_steps=50, X=self.X, Y=self.Y)
            self.rec['cap_bal'][i_p,idx] = cap
            cap, _ = self.get_tolerance('gauss', p*3, n_tests=20, n_steps=50, X=self.X, Y=self.Y)
            self.rec['cap_synz'][i_p,idx] = cap
            cap, _ = self.get_tolerance('gauss1', p*4, n_tests=20, n_steps=50, X=self.X, Y=self.Y)
            self.rec['cap_syn1'][i_p,idx] = cap
        self.dynamic_inh = True
        self.U = U_backup.clone()
        self.theta = theta_backup.clone()
        self.compute_W()

    
    def normalize(self):
        if self.norm_type == 'u':
            #U_factor =  torch.maximum(self.one, torch.linalg.norm(self.U.flatten(), ord=2)/self.length)
            #U_small = self.U#*self.Mask_small
            U_factor = self.U.pow(2).sum(dim=(1,2), keepdim=True)/self.length/self.U.shape[2]
            #self.theta /= U_factor
            self.U /= U_factor.pow(0.5)#*self.Mask_small + self.Mask_big
            #self.U0 /= U_factor.pow(0.5)#*self.Mask_small + self.Mask_big
        
        else:
            #w_factor =  torch.maximum(self.one, torch.linalg.norm(self.w, ord=self.norm_type)/self.length)
            #W_small = self.W#*self.Mask_small
            W_factor = self.W.pow(self.norm_type).sum(dim=1, keepdim=True)/self.length
            #self.theta /= W_factor
            self.U /= W_factor.pow(1/(self.norm_type*self.z))#*self.Mask_small + self.Mask_big
            #self.U0 /= W_factor.pow(1/(self.norm_type*self.z))#*self.Mask_small + self.Mask_big

        self.compute_W()


    def compute_masks(self):
        torch.gt(self.W, 1e-10, out=self.Mask_bool)
        torch.mul(self.Mask_bool, self.ones, out=self.Mask_big)
        torch.sub(self.ones, self.Mask_big, out=self.Mask_small)

    
    def reinitialize(self):
        self.U *= self.Mask_big # remove small weights
        self.U += torch.normal(mean=self.U0, std=self.U0/100)*self.Mask_small # reinitialize small weights
        self.compute_W()


    def print_progress(self, day, epoch, interval):
        if epoch % interval == 0:
            #X = self.X_lst[day]
            #Y = self.Y_lst[day]
            error_type = 'bernoulli-balanced'
            print(f'Day: {day:3}/{self.n_days}, '+
                  f'Epoch: {epoch:7}, '+
                  f'theta: {self.get_theta().item():.5}, '+
                  f'd_theta: {self.d_theta.item():.5}, '+
                  f'bias: {self.get_bias().mean().item():.5}, '+
                  f'l0: {self.get_l0()[0].item():.5}, '+
                  f'l1: {self.get_l1()[0].item():.5}, '+
                  f'l2: {self.get_l2()[0].item():.5}, '+
                  f'err: {self.get_error().item():.5}, '+
                  f'RR: {self.get_capacity().item():.5}, '+
                  f'rho: {self.get_rho()[0].item():.5}, '+
                  f'margin: {self.get_margin()[0].item():.5}')


    def optimize(self):
        if self.optimizer_type == 'bp':
            order = torch.zeros(self.M_day, device=self.device, dtype=int)
            H_ = torch.zeros((self.N,self.M_day), device=self.device)
            Mask_mu = torch.zeros((self.N,self.M_day), device=self.device)
            N_vec = torch.arange(self.N, device=self.device)
            epoch_global = 0

            if self.record_on:
                self.record_epoch(0)

            # Loop over days
            for day in range(self.n_days):
                # Print awake message
                print(f'------------------------- AWAKE -----------------------------')

                # Set the weight mask, reinitialize new weights, and normalize
                self.compute_masks()
                #self.reinitialize()
                self.clip()
                #self.normalize()
                self.theta = self.get_bias()

                # Extract new patterns for awake state
                X_wake = self.X_lst[day]
                Y_wake = self.Y_lst[day]

                #self.rec_tol(0)
              
                # Loop over awake state epochs
                for epoch_wake in range(self.n_epochs_wake):
                    self.print_progress(day, epoch_wake, interval=1)
                    #print(self.get_tolerance('bernoulli-balanced', 0, 20, 20)[0].item())
                    # break loop if all patterns encoded
                    #if self.get_error(X_wake, Y_wake) == 0:
                    #    break

                    # draw random pattern order
                    torch.randperm(self.M_day, out=order)

                    # Loop over patterns in order
                    for mu in order:
                        X_mu = X_wake[:,mu:mu+1]
                        Y_mu = Y_wake[:,mu:mu+1]

                        # keep track of epochs
                        epoch_global += 1
                        #self.print_progress(day, epoch_wake, interval=1)

                        # One-shot learning
                        self.compute_dWdU()
                        self.compute_masks()
                        self.forward_(X_mu, T=1)
                        #print((S==X_wake[:,mu:mu+1]).all())
                        #Eth = torch.heaviside(self.d_theta.abs(), self.zero)
                        Eth = self.d_theta.abs().pow(1)
                        E = torch.heaviside(self.get_error(X_mu, Y_mu)/self.f, self.zero)
                        #print(f'E: {E}, Eth: {Eth.item()}')
                        if self.mem_type=='attractor':
                            self.U[:,:,0] += self.eta_wake*( Eth*torch.matmul(X_mu-self.f, X_mu.T-self.f) )#*self.dWdU[:,:,0]#/(0.1**0.5)
                                                     #)#+ 5*(1 - self.W.sum(dim=1, keepdim=True)/self.length)*self.U*self.U0 )
                        elif self.mem_type=='sequence':
                            self.U[:,:,0] += self.eta_wake*( Eth*torch.matmul((Y_mu*0.5+0.5)-self.f, X_mu.T-self.f))#*self.dWdU[:,:,0]/(0.1**0.5)

                        self.clip()
                        #self.U[:,:,0] /= ( self.U.pow(2).sum(dim=(1,2))/self.length/self.U.shape[2] ).pow(0.5)[:,None]
                        self.compute_W()
                        #self.normalize()

                        # Update threshold
                        if self.with_bias:
                            #self.theta -= self.eta_theta*torch.sign(self.theta - self.get_bias())
                            #self.theta -= self.eta_theta*Y_mu.flatten()
                            self.theta = self.get_bias()

                        if self.record_on:
                            self.record_epoch(epoch_global)
                            self.rec['replays'][mu,0] += Eth[0]
                            self.rec['ymin'][:,epoch_global] = Y_mu[-30:,0].clone()
                            self.rec['t'][epoch_global] = self.rec['t'][epoch_global-1] + 1/self.M_day

                # Extract patterns for sleep state
                print(((self.Y*(self.W@self.X - self.theta[:,None])).min(dim=0)[0]>0).sum())
                idx_attr = ((self.Y*(self.W@self.X - self.theta[:,None])).min(dim=0)[0]>0)
                self.X = self.X[:,idx_attr]
                self.Y = self.Y[:,idx_attr]
                self.M = idx_attr.sum()
                self.M_day = self.M_day*self.n_days
                mu_first = self.M_day*day # change this to zero to replay all past patterns
                mu_last = self.M_day*(day+1)
                X_sleep = self.X[:,mu_first:mu_last]
                Y_sleep = self.Y[:,mu_first:mu_last]
                X_wake = X_sleep.clone()
                Y_wake = Y_sleep.clone()
                self.noise = self.noise[:,:self.M]
                self.rec['radius_min'] = self.rec['radius_min'][:self.M,:]
                self.rec['radius_avg'] = self.rec['radius_avg'][:self.M,:]

                # Print pre-sleep performance
                print(f'error_day: {self.get_error(X_wake, Y_wake).item()}, '+
                      f'margin_day: {self.get_margin(X_wake, Y_wake)[0].item()}, '+
                      f'error_all: {self.get_error(X_sleep, Y_sleep).item()}, '+
                      f'margin_all: {self.get_margin(X_sleep, Y_sleep)[0].item()}')

                # Print asleep message
                print(f'------------------------- ASLEEP -----------------------------')

                # record weights pre sleep
                #self.rec['w'][:,0] = self.W.flatten()
                self.rec['theta_all'][:,0] = self.theta.clone()
                self.rec['radius_min'][:,0] = self.get_min_radius()
                self.rec['radius_avg'][:,0] = self.get_avg_radius()
                self.rec_tol(1)

                #H_means = (self.W@X_sleep - self.theta[:,None]).abs().mean(dim=1, keepdim=True)

                # Loop over epochs in sleep state
                for epoch_sleep in range(self.n_epochs_sleep):
                    epoch_global += 1
                    self.print_progress(day, epoch_sleep, interval=1e3)

                    if epoch_sleep%100==0 and epoch_sleep<3000:
                        self.rec['w'][:,int(epoch_sleep/100)] = self.W.clone().flatten()
                        if epoch_sleep==0:
                            ufac = ( self.U.pow(2).sum(dim=(1,2), keepdim=True)/self.length/self.U.shape[2] )
                            self.U /= ufac.pow(0.5); self.compute_W()
                            self.theta /= ufac[:,0,0].pow(0.5).pow(self.z)
                            H_means = (self.W@X_sleep - self.theta[:,None]).abs().mean(dim=1, keepdim=True)

                    if self.exact_mode:
                        # Find weakest weights
                        torch.matmul(self.W, X_wake, out=H_)
                        torch.sub(H_, self.theta[:,None], out=H_)
                        Y_self = torch.sign(H_)
                        #torch.mul(H_, Y_wake, out=H_)
                        torch.mul(H_, Y_self, out=H_)
                        mu_min = torch.argmin(H_, dim=1)
                        Mask_mu.fill_(0.)
                        Mask_mu[N_vec, mu_min] += 1.
                        X_min = torch.matmul(Mask_mu, X_wake.T)
                        y_min = Y_self[N_vec, mu_min]
                        #y_min = torch.sign((self.W@X_min.T).diag() - self.theta) # self-supervised

                        # Update weights
                        self.compute_dWdU()
                        self.compute_masks()
                        self.eta_sleep_ = self.eta_sleep + (5e-2 - self.eta_sleep)*(1-np.exp(-epoch_sleep/1e2))
                        #self.eta_sleep_ = self.eta_sleep + (5e-3 - self.eta_sleep)*(1-np.exp(-epoch_sleep/1e2))
                        self.U += self.eta_sleep_*y_min[:,None,None]*X_min[:,:,None]*self.dWdU*self.Mask_big[:,:,None] #*self.Mask_small
                        #self.U += 0.5*self.eta_sleep_*y_min[:,None]*self.dWdU
                        self.clip()
                        self.normalize()

                    else:
                        torch.matmul(self.W, X_sleep, out=H_)
                        torch.sub(H_, self.theta[:,None], out=H_)
                        Y_self = torch.sign(H_)
                        #torch.mul(H_, Y_self, out=H_)
                        #Loss = torch.exp(-self.beta*H_.abs()/H_.abs().mean(dim=1, keepdim=True))
                        Loss = torch.exp(-self.beta*H_.abs()/H_means)
                        H_means = H_.abs().mean(dim=1, keepdim=True) #(self.W@X_sleep - self.theta[:,None]).abs().mean(dim=1, keepdim=True)
                        Mask_mu = (Loss*Y_self) / Loss.sum(dim=1, keepdim=True)
                        dLdW = torch.matmul(Mask_mu, X_sleep.T)
                       
                        self.compute_dWdU()
                        self.compute_masks()
                        #eta_normer = Loss.sum(dim=1).max()/10 #1e-1 if self.beta==10 else 1e-4
                        if self.z==1: self.eta_sleep_ = self.eta_sleep #+ (4e-2 - self.eta_sleep)*(1-np.exp(-epoch_sleep/2e1))
                        else: self.eta_sleep_ = self.eta_sleep + (40e-2 - self.eta_sleep)*(1-np.exp(-epoch_sleep/4e1))
                        #self.eta_sleep_ /= eta_normer
                        #self.eta_sleep_ = self.eta_sleep + (5e-3 - self.eta_sleep)*(1-np.exp(-epoch_sleep/1e2))
                        self.U += self.eta_sleep_ * dLdW[:,:,None] * self.dWdU * self.Mask_big[:,:,None] #*self.Mask_small
                        #self.U += 0.5*self.eta_sleep_*y_min[:,None]*self.dWdU
                        self.clip()
                        self.normalize()

                    if epoch_sleep<499:
                        self.rec['radius_min'][:,epoch_sleep+1] = self.get_min_radius()
                        self.rec['radius_avg'][:,epoch_sleep+1] = self.get_avg_radius()

                    #if epoch_sleep == 200:
                    #    print(self.get_margin()[0].item())
                    #    self.rec_tol(2)
                    #    print('tol 3 done!')
                    #    print(self.get_margin()[0].item())

                    #if epoch_sleep > 1e4:
                    #    self.eta_sleep = 10e-3

                    # Update threshold
                    if self.with_bias:
                        if self.exact_mode:
                            self.eta_theta_ = self.eta_theta#*np.exp(-10*epoch/self.n_epochs) # for quenching eta_theta
                            torch.matmul(self.W, X_wake, out=H_)
                            H_ = torch.sort(H_, dim=1)[0]
                            idx_low = (H_-self.theta[:,None]).sign().diff(dim=1).argmax(dim=1)
                            idx_high = idx_low + 1
                            th_low = H_[N_vec, idx_low]
                            th_high = H_[N_vec, idx_high]
                            self.theta = (th_low + th_high)/2
                            self.theta += th_low.normal_(std=1e-5)

                        else:
                            #idx_low = torch.sort(H_-self.theta[:,None], dim=1)[0].sign().diff(dim=1).argmax(dim=1)
                            #idx_high = idx_low + 1
                            #th_low = torch.sort(H_, dim=1)[0][N_vec, idx_low]
                            #th_high = torch.sort(H_, dim=1)[0][N_vec, idx_high]
                            #self.theta = (th_low + th_high)/2
                            #   Mask_H_ = (Loss*H_) / Loss.sum(dim=1, keepdim=True)
                            #   self.theta += Mask_H_.mean(dim=1)[0]
                            y_min = Mask_mu.sum(dim=1)
                            self.theta -= self.eta_sleep_*y_min
                            #self.theta -= self.eta_theta_*torch.sign(self.theta - self.get_bias())
                            #self.theta = self.get_bias()

                    if self.record_on:
                        self.record_epoch(epoch_global)
                        #self.rec['replays'][mu_min,1] += 1
                        self.rec['ltp-ratio_pop'][epoch_global] = (0.5*y_min + 0.5).mean()
                        self.rec['ltp-ratio_time'] += (0.5*y_min + 0.5)
                        self.rec['ymin'][:,epoch_global] = y_min[-30:].clone()
                        self.rec['t'][epoch_global] = self.rec['t'][epoch_global-1] + 1
                
                # Print post-sleep performance
                print(f'error_day: {self.get_error(X_wake, Y_wake).item()}, '+
                      f'margin_day: {self.get_margin(X_wake, Y_wake)[0].item()}, '+
                      f'error_all: {self.get_error(X_sleep, Y_sleep).item()}, '+
                      f'margin_all: {self.get_margin(X_sleep, Y_sleep)[0].item()}')

                # record weights post sleep
                self.rec['w'][:,-1] = self.W.clone().flatten()
                self.rec['theta_all'][:,1] = self.theta.clone()
                self.rec['radius_min'][:,-1] = self.get_min_radius()
                self.rec['radius_avg'][:,-1] = self.get_avg_radius()
                self.rec['ksi'] = self.X.clone()
                self.rec_tol(3)
