# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pennylane as qml
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets

import numpy as pure_np
from pennylane import numpy as np


import os 
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from wp_TK import tk_lib as tk

import matplotlib.pyplot as plt

import tqdm
from itertools import product

from pennylane_cirq import ops as cirq_ops
import nk_lib
np.random.seed(42+1)
# -

# # Some global variables (used below!)

features = 2
width = 5
depth = 8 

# +

dev = qml.device("default.qubit", wires=width)
wires = list(range(width))

# init the embedding kernel
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x, params, wires), dev)
# -

# # Dataset

# +
dim = 4

init_false = False
init_true = False
for i in range(dim):
    for j in range(dim):
        pos_x = i
        pos_y = j
        data = (np.random.random((40,2))-0.5)/(2*dim)
        data[:,0] += (2*pos_x+1)/(2*dim)
        data[:,1] += (2*pos_y+1)/(2*dim)
        if (i%2 == 0 and j%2 == 0) or (i%2 == 1 and j%2 == 1):
            if init_false == False:
                false = data
                init_false = True
            else:
                false = np.vstack([false, data])
        else:
            if init_true == False:
                true = data
                init_true = True
            else:
                true = np.vstack([true, data])
print(false.shape)
print(true.shape)

# +
samples = 30 

np.random.shuffle(false)
np.random.shuffle(true)

X_train = np.vstack([false[:samples], true[:samples]])
y_train = np.hstack([-np.ones((samples)), np.ones((samples))])
X_test = np.vstack([false[samples:2*samples], true[samples:2*samples]])
y_test = np.hstack([-np.ones((samples)), np.ones((samples))])

print("The training data is as follows:")
plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker=".", label="train, 1")
plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker=".", label="train, -1")
print("The test data is as follows:")
plt.scatter(X_test[np.where(y_train == 1)[0],0], X_test[np.where(y_train == 1)[0],1], color="b", marker="x", label="test, 1")
plt.scatter(X_test[np.where(y_train == -1)[0],0], X_test[np.where(y_train == -1)[0],1], color="r", marker="x", label="test, -1")
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.legend()
plt.show()


# -

# # Pretrained optimal parameters

# +
def random_params(num_wires, num_layers):
    return np.random.uniform(0, 2*np.pi, (num_layers, 2, num_wires))

params = random_params(width, depth)
# -

print(params)


# # Rigetti Circuit

# +
def rigetti_layer(x, params, wires, i0=0, inc=1):
    i = i0
    N = len(wires)
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
    for j, wire in enumerate(wires):    
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
    for j, wire in enumerate(wires):
        qml.RY(params[0, j], wires=[wire])
        
    for j in range(N//2):
        qml.CRZ(params[1][2*j], wires=wires[2*j:2*(j+1)])
    for j in range(N//2):
        qml.CRZ(params[1][2*j+1], wires=[wires[2*j+1], wires[(2*j+2)%N]])
    if N%2==1: # There is the "hop the boundary" gate missing if N is odd.
        qml.CRZ(params[1][-1], wires=[wires[-1], wires[0]])


@qml.template
def rigetti_ansatz(x, params, wires):
    for j, layer_params in enumerate(params):
        rigetti_layer(x, layer_params, wires, i0=j * len(wires))


# -

# #  Define noise

def noise_channel(base_p, data=None, wires=None):
    if data is None:
        # Idling gate
        angle = np.pi
    elif len(data)==0:
        # Non-parametrized gate ( Hadamard in this circuit )
        angle = 3 * np.pi / 2
    else:
        # Parametrized gate
        angle = data[0]
        
    relative_p = base_p * ( np.abs(angle) / (2*np.pi) )
    noise_ops = [cirq_ops.Depolarize(relative_p, wires=wire) for wire in wires]
    return noise_ops


# # Simulation

# +
fix_diag = False # Compute the diagonal entries for mitigation.
rigetti_ansatz_mapped = lambda x, params: rigetti_ansatz(x, params, range(width))

shot_numbers = [10]#100]#, 1000, 10000, 0]
# shot_numbers = [10]
noise_probabilities = np.arange(0.05, 0.06, 0.005)
print(noise_probabilities)
kernel_matrices = {}
for noise_p, shots in tqdm.notebook.tqdm(product(noise_probabilities, shot_numbers)):
    analytic_device = (shots==0)
    shots_device = 1 if shots==0 else shots # shots=0 raises an error...

    dev = qml.device("cirq.mixedsimulator", wires=width, shots=shots_device, analytic=analytic_device)
    k = nk_lib.noisy_kernel(
        rigetti_ansatz_mapped,
        dev,
        noise_channel=noise_channel,
        args_noise_channel=(noise_p,),
        noise_application_level='per_gate',
    )
    k_mapped = lambda x1, x2: k(x1, x2, params)
    
    K = qml.kernels.square_kernel_matrix(X_train, k_mapped, assume_normalized_kernel=fix_diag)       
    kernel_matrices[(float(noise_p), shots)] = K

# -

print(dev.num_executions)

print(kernel_matrices)


