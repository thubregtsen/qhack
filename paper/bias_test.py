# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Noisy circuit simulations on Checkerboard dataset: Data Generation
# This notebook simulates a noisy device with the noise model described in the appendix.
# Here we compute the kernel matrices for the different base noise constants and numbers of measurements,
# the processing can be found in a separate notebook. 
#
# ### Note that this notebook takes a signficant amount of time to run (hours)

# + tags=[]
import time
import datetime
from functools import partial

import matplotlib.pyplot as plt
import pennylane as qml
import numpy as pure_np
from pennylane import numpy as np
from pennylane_cirq.ops import Depolarize
from dill import dump, load
import multiprocessing
import src.kernel_helper_functions as khf
from src.datasets import checkerboard
# -


# # Some global variables (used below!)

# + tags=[]
num_wires = 5
use_trained_params = False

sub_filename = f'data/noisy_sim/sub_kernel_matrices_Checkerboard_{"" if use_trained_params else "un"}trained.dill'
filename = f'data/noisy_sim/kernel_matrices_Checkerboard_{"" if use_trained_params else "un"}trained.dill'
# -

# # Checkerboard dataset

# + tags=[]
np.random.seed(43)
X_train, y_train, X_test, y_test = checkerboard(30, 30, 4, 4)

print("The training data is as follows:")
plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker=".", label="train, 1")
plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker=".", label="train, -1")
print("The test data is as follows:")
plt.scatter(X_test[np.where(y_train == 1)[0],0], X_test[np.where(y_train == 1)[0],1], color="b", marker="x", label="test, 1")
plt.scatter(X_test[np.where(y_train == -1)[0],0], X_test[np.where(y_train == -1)[0],1], color="r", marker="x", label="test, -1")
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.legend()

X = X_train

opt_param = np.tensor([[[ 6.22961793,  6.1909463 ,  6.24821366,  1.88800397,  1.6515437 ],
                        [-3.50578116,  2.87429701, -0.55558014, -2.97461847,  4.3646466 ]],
                       [[ 4.59893525, -0.01877453,  4.86909045,  1.61046237,  4.3342154 ],
                        [ 6.54969706,  0.76974914,  6.13216135,  3.19770538,  0.35820405]],
                       [[-0.06825097,  5.46138114, -0.38685812,  2.62531926,  5.94363286],
                        [ 3.84330489,  7.62532526,  3.31992264,  4.53318486,  2.90021471]],
                       [[ 3.27271762,  6.284331  , -0.0095848 ,  1.71022713,  1.72119449],
                        [ 5.26413732, -0.5363315 ,  0.02694912,  1.85543017,  0.09469438]],
                       [[ 1.61977233,  2.12403094,  1.52887576,  1.87843468,  5.10722657],
                        [ 1.83547388,  0.10519713, -0.14516422,  2.34971729, -0.15396484]],
                       [[ 1.15227788,  4.42815449,  4.77992685,  2.00495827,  4.68944624],
                        [ 1.90477385, -0.22817579,  6.21664772,  0.34922366,  6.44687527]],
                       [[ 4.47834114,  5.80827321,  4.8221783 ,  2.07389821,  0.40258912],
                        [ 6.07380714,  6.33676481,  6.17787822,  1.86149763,  6.59189267]],
                       [[ 5.56242829,  4.49153866,  3.66496649,  4.76465886,  0.80552847],
                        [ 3.36765317,  3.41585518,  1.40441779,  1.24372229,  5.85030332]]], requires_grad=True)
if use_trained_params:
    param = np.copy(opt_param)
    print(f"Using trained parameters with average magnitude {np.mean(np.abs(param))}")
else:
    np.random.seed(43)
    param = np.random.random(size=opt_param.shape) * 4 * np.pi - 2 * np.pi
    print(f"Using untrained parameters with average magnitude {np.mean(np.abs(param))}")


# -

# # Rigetti Circuit

# + tags=[]
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


def rigetti_ansatz(x, params, wires):
    with qml.tape.QuantumTape() as tape:
        for j, layer_params in enumerate(params):
            rigetti_layer(x, layer_params, wires, i0=j * len(wires))
    return tape


# -

# # Simulate noise

# ## Circuit Level Noise single qubit depolarizing channel

# + tags=[]
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
    noise_ops = [Depolarize(relative_p, wires=wire) for wire in wires]
    return noise_ops


# -

# # Noise mitigation computations
#
# ## Compute noisy kernel matrix

# + tags=[]
# Parallelization of the previous cell
rigetti_ansatz_mapped = partial(rigetti_ansatz, wires=range(num_wires))

shot_numbers = [10, 30, 100, 300, 1000, 3000]
noise_probabilities = [0.01*i for i in range(11)]
noise_probabilities = [0.001*i for i in range(1, 4)]
# noise_probabilities = [0.0]
num_reps = 200

# + tags=[]
my_time = lambda *args: ":".join(str(datetime.datetime.now()).split(':')[:-1])

def job(bnr, shots, num_reps):
    orig_shots = shots
    if shots == 0:
        num_batches = 1
        device_shots = None
    else:
        shots = sum([[s] * num_reps for s in shots], start=[])
        num_batches = len(shots)
        device_shots = shots
    
    if bnr==0.:
        dev = qml.device("lightning.qubit", wires=num_wires, shots=device_shots)
        noise_application_level = "never"
    else:
        dev = qml.device("cirq.mixedsimulator", wires=num_wires, shots=device_shots)
        noise_application_level = 'per_gate'
        
    k = khf.noisy_kernel(
        rigetti_ansatz_mapped,
        dev,
        noise_channel=noise_channel,
        args_noise_channel=(bnr,),
        noise_application_level=noise_application_level,
    )
    k_mapped = partial(k.circuit, params=param)
    K = khf.square_kernel_matrix_batched(X, k_mapped, assume_normalized_kernel=False, num_batches=num_batches)
    return K

start = time.time()

def run(bnr):
    print(f"Starting run with base noise rate {bnr} ({my_time()})\n")
    sub_start = time.process_time()
    sub_kernel_matrices = {}
    
    print(f"Starting job with base noise rate {bnr} without shots ({my_time()})\n")
    # Run the shot-noise-free job
    K_no_shots = job(bnr, 0, num_reps)
    # and store it
    sub_kernel_matrices[(float(bnr), 0)] = K_no_shots
    
    print(f"Starting job with base noise rate {bnr}, with shots and num_reps={num_reps} ({my_time()})\n")
    # Run the shot-noisy job
    K = job(bnr, shot_numbers, num_reps)
    # and extract all repetition experiments from it
    for shots_idx, s in enumerate(shot_numbers):
        for r in range(num_reps):
            sub_kernel_matrices[(float(bnr), s, r)] = K[shots_idx*num_reps+r]
    
    sub_pure_np_kernel_matrices = {key: pure_np.asarray(mat) for key, mat in sub_kernel_matrices.items()}
    sub_fn = f"{sub_filename.split('.')[0]}_{float(bnr)}_{str([0]+shot_numbers).replace(' ', '').replace(',', '_')}_{num_reps}.dill"
    dump(sub_pure_np_kernel_matrices, open(sub_fn, 'wb+'))
    print(f"Base noise rate {bnr}: {(time.process_time()-sub_start)/60:.3f} minutes")


pool = multiprocessing.Pool()
pool.map(run, noise_probabilities)
print(f"{(time.time()-start)/60} minutes")
# -

# # Save data

# + tags=[]
# Merge matrix sets
kernel_matrices = {}
for bnr in noise_probabilities:
    sub_fn = f"{sub_filename.split('.')[0]}_{float(bnr)}_{str([0]+shot_numbers).replace(' ', '').replace(',', '_')}_{num_reps}.dill"
    try:
        sub_mats = load(open(sub_fn, 'rb+'))
        kernel_matrices.update(sub_mats)
    except:
        print(sub_fn)
    

# pure_np_kernel_matrices = {key: pure_np.asarray(mat) for key, mat in kernel_matrices.items()}
# dump(pure_np_kernel_matrices, open(filename, 'wb+'))
# + tags=[]
target = np.outer(y_train, y_train)
TA = {}
A = {}
no_noise_mat = kernel_matrices[(0.0, 0)]
for bnr in noise_probabilities:
    no_shot_mat = kernel_matrices[(bnr, 0)]
    for shots in shot_numbers:
        these_mats = [kernel_matrices[(bnr, shots, i)] for i in range(num_reps)]
        these_TA = [qml.math.frobenius_inner_product(mat, target, normalize=True) for mat in these_mats]
        these_A = [qml.math.frobenius_inner_product(mat, no_noise_mat, normalize=True) for mat in these_mats]
        TA[(bnr, shots)] = qml.math.stack(these_TA)
        A[(bnr, shots)] = qml.math.stack(these_A)
    TA[(bnr, 0)] = qml.math.frobenius_inner_product(no_shot_mat, target, normalize=True)
    A[(bnr, 0)] = qml.math.frobenius_inner_product(no_shot_mat, no_noise_mat, normalize=True)

# + tags=[]
show = "alignment"
show = "target alignment"

if show=="alignment":
    D = A
    label = "A(K, $K^{\mathrm{(dev)}}$)"
elif show=="target alignment":
    D = TA
    label = "TA($K^{\mathrm{(dev)}}$)"
    
num_bnrs = len(noise_probabilities)
fig, axs = plt.subplots(num_bnrs, 2, figsize=(15, num_bnrs*5))
for bnr, ax in zip(noise_probabilities, axs):
    for shots in shot_numbers:
        _ta = D[(bnr, shots)]
        ax[0].hist(_ta, alpha=0.5, label=f"{shots} Shots")
        ax[1].errorbar(shots, np.mean(_ta), np.std(_ta), c='r', ecolor='b', capsize=6., marker='o')
    # print([np.mean(D[(bnr, shots)]).item() for shots in shot_numbers])
    _ta = D[(bnr, 0)]
    ax[0].plot([_ta]*2, ax[0].get_ylim(), c='k', label="No shot noise")
    ax[1].plot([min(shot_numbers), max(shot_numbers)], [_ta]*2, c='k')
    ax[0].legend(title=f"bnr={bnr}")
    ax[0].set(xlabel=label, ylabel="frequency")
    ax[1].set(xlabel="Shots", ylabel=label, xscale="log")

# + tags=[]
shots = [10, 10]
dev = qml.device("cirq.mixedsimulator", wires=3, shots=shots)
print(dev.shots)

@qml.qnode(dev)
def qnode(x):
    [qml.RX(x, wires=w) for w in dev.wires]
    return qml.expval(qml.PauliZ(0))

qnode(0.3)
# -


