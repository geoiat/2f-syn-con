import numpy as np
from datetime import datetime

#############################################################################################
###############      Parameter definitions     ##############################################
#############################################################################################

N = 1000
dt = 5e-3
#n_steps = 100000
seed = 1
w0 = 1

T = 999.
T_sample = 1

# Book-keeping
n_steps = int(T/dt)
n_samples = int(T/T_sample)
steps_per_sample = int(T_sample/dt)

# Storage
z_vec = np.array([1, 2, 3, 4, 5], dtype=int)
models = [1, 2, 3, 4, 5, 10, 'lou', 'kes']

# random number generator
rng = np.random.default_rng(seed)

# Auxiliary variables
X = np.zeros((N,2))

# general noise vector
noise = np.zeros((N,2)) #rng.normal(size=(N,T))
sgn_noise = np.zeros((N,2))
abs_noise = np.zeros((N,2))

# hadamard product parameterization stochastic process
sig_u = 0.1*0.5
tau_u = 3e1
dp = 0.01
k = 0.5
bias = 0.1


U = {1: np.ones((N,1))*w0**(1/1),
     2: np.ones((N,2))*w0**(1/2),
     3: np.ones((N,3))*w0**(1/3),
     4: np.ones((N,4))*w0**(1/4),
     5: np.ones((N,5))*w0**(1/5),
     10: np.ones((N,10))*w0**(1/10)
    }

w = {1: np.ones(N)*w0,
     2: np.ones(N)*w0,
     3: np.ones(N)*w0,
     4: np.ones(N)*w0,
     5: np.ones(N)*w0,
     10: np.ones(N)*w0,
     'lou': np.ones(N),
     'kes': np.ones(N),
    }

side = {1: np.ones(N),
        2: np.ones(N),
        3: np.ones(N),
        4: np.ones(N),
        5: np.ones(N),
        10: np.ones(N),
        }

w_mat = {1: np.zeros((N,n_samples)),
         2: np.zeros((N,n_samples)),
         3: np.zeros((N,n_samples)),
         4: np.zeros((N,n_samples)),
         5: np.zeros((N,n_samples)),
         10: np.zeros((N,n_samples)),
         'lou': np.zeros((N,n_samples)),
         'kes': np.zeros((N,n_samples)),
        }

mean_vec = {1: np.zeros((n_samples)),
            2: np.zeros((n_samples)),
            3: np.zeros((n_samples)),
            4: np.zeros((n_samples)),
            5: np.zeros((n_samples)),
            10: np.zeros((n_samples)),
            'lou': np.zeros(n_samples),
            'kes': np.zeros(n_samples),
           }

std_vec = {1: np.zeros((n_samples)),
           2: np.zeros((n_samples)),
           3: np.zeros((n_samples)),
           4: np.zeros((n_samples)),
           5: np.zeros((n_samples)),
           10: np.zeros((n_samples)),
           'lou': np.zeros(n_samples),
           'kes': np.zeros(n_samples),
          }

num_vec = {1: np.zeros((n_samples)),
           2: np.zeros((n_samples)),
           3: np.zeros((n_samples)),
           4: np.zeros((n_samples)),
           5: np.zeros((n_samples)),
           10: np.zeros((n_samples)),
           'lou': np.zeros(n_samples),
           'kes': np.zeros(n_samples),
          }

idx_alive = {1: np.ones(N, dtype=bool),
             2: np.ones(N, dtype=bool),
             3: np.ones(N, dtype=bool),
             4: np.ones(N, dtype=bool),
             5: np.ones(N, dtype=bool),
             10: np.ones(N, dtype=bool),
            'lou': np.ones(N, dtype=bool),
            'kes': np.ones(N, dtype=bool),
            }

idx_dead = {1: np.zeros(N, dtype=bool),
            2: np.zeros(N, dtype=bool),
            3: np.zeros(N, dtype=bool),
            4: np.zeros(N, dtype=bool),
            5: np.zeros(N, dtype=bool),
            10: np.zeros(N, dtype=bool),
            'lou': np.zeros(N, dtype=bool),
            'kes': np.zeros(N, dtype=bool),
           }


#############################################################################################
###################      Simulation     #####################################################
#############################################################################################

for t in range(n_steps):
    t_now = t*dt
    if t_now%1e2 == 0:
        print(t_now)

    #age_vec += 1
    rng.standard_normal(out=noise)
    # rng.random(out=sgn_noise)
    # np.abs(noise, out=abs_noise)

    for z in z_vec:
        homeo = (1 - np.mean(w[z][idx_alive[z]]**(2/z))/w0)
        fast_U = U[z][:,0]
        slow_U = np.product(U[z][:,1:], axis=1)
        U[z][:,0] += sig_u*((1-k)*noise[:,0] + k*slow_U*noise[:,1])*np.sqrt(dt) + bias*sig_u*((1-k) + k*slow_U)*dt + 10*homeo*fast_U*dt #+ 0.1*sig_u*np.sign(np.median(w[z]) - w[z])*dt # hadamard parameterization
        U[z][:,1:] -= np.diff(U[z][:,0:2], axis=1)*dt/tau_u - 10*homeo*U[z][:,1:]*dt/tau_u #+ penalty*U[:,1:]*dt/tau_u + 0.1*noise[:,1]*np.sqrt(dt/tau_u) + 0.1*dt
        U[z][idx_dead[z],:] = 0
        np.clip(U[z], a_min=0, a_max=None, out=U[z])
        np.product(U[z], axis=1, out=w[z])

    for key in models:
        idx_alive[key] = w[key]>0
        idx_dead[key] = w[key]<=0

    if t%steps_per_sample == 0:
        i = int(t/steps_per_sample)

        for key in models:
            w_mat[key][:,i] = w[key]
            mean_vec[key][i] = w[key][idx_alive[key]].mean()
            std_vec[key][i] = w[key][idx_alive[key]].std()
            num_vec[key][i] = idx_alive[key].mean()



#############################################################################################
###################        Save         #####################################################
#############################################################################################

data = {}
data['date'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
data['models'] = models
data['w'] = w_mat
data['mean'] = mean_vec
data['std'] = std_vec
data['num'] = num_vec
data['dt'] = dt
data['k'] = k
data['bias'] = bias
data['z'] = z_vec
data['sig'] = sig_u
data['tau'] = tau_u

np.save('./data/syn_noise.npy', data)
