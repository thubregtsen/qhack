# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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
import numpy as np
import pandas as pd
# import torch
# from torch.nn.functional import relu

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor
import pennylane.kernels as kern

import multiprocessing as mproc

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

# import jupytext # this is not really needed :-)
from cake_dataset import Dataset as Cake
import tk_lib
import time

np.random.seed(42)
# -


# # Data


data = 'tom'
if data=='iris':
    # load the data
    X, y = load_iris(return_X_y=True) 

    print("The dataset contains X and y, each of length", len(X))
    print("X contains", len(X[0]), "features")
    print("y contains the following classes", np.unique(y))

    # pick inputs and labels from the first two classes only,
    # corresponding to the first 100 samples
    # -> meanig y now consists of 2 classes: 0, 1; still stored in order, balanced 50:50
    X = X[:100:4,:2]
    y = y[:100:4]

    print("The dataset is trimmed so that the total number of samples are ", len(X))
    print("The original tutorial sticked with 4 features, I (Tom) reduced it to ", len(X[0]))

    # scaling the inputs is important since the embedding we use is periodic
    # -> data is scaled to np.min(X)=-2.307; np.max(X)= 2.731
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    print("X is normalized to the range", np.min(X_scaled), np.max(X_scaled))

    # scaling the labels to -1, 1 is important for the SVM and the
    # definition of a hinge loss
    # -> now making the 2 classes: -1, 1
    y_scaled = 2 * (y - 0.5)
    print("y is normalized to drop a class, and now contains", np.sum([1 if x==-1 else 0 for x in y_scaled]), "\"-1\" classes and ", np.sum([1 if x==1 else 0 for x in y_scaled]), "\"1\" classes")

    # -> result of train_test_split:
    # len(X_train)=75, 39 labelled 1, 36 labelled -1
    # len(X_test)=25
    # data is shuffled prior to split (shuffled variable in train_test_split is default True)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)
    print("Lastly, the data is shuffled and split into", len(X_train), "training samples and", len(X_test), "samples")

    print("The training data is as follows:")
    plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", label=1)
    plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", label=-1)
    plt.legend()
elif data=='cake':
    num_sectors = 10
    num_samples = 0
    cake = Cake(num_samples, num_sectors)
    X = np.array([[x, y] for x, y in zip(cake.X, cake.Y)])
#     print(X)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    Y_scaled = cake.labels_sym
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled)
    plt.scatter(X_train[y_train == 1,0], X_train[y_train == 1,1], color="b", label=1)
    plt.scatter(X_train[y_train == -1,0], X_train[y_train == -1,1], color="r", label=-1)
    plt.legend()
elif data=='tom':
    dataset_index = 7 # range(9)
    train_data = pd.read_pickle("../plots_and_data/train.txt")
    test_data = pd.read_pickle("../plots_and_data/test.txt")
    X_train, y_train = train_data.iloc[dataset_index]
    X_test, y_test = test_data.iloc[dataset_index]

# # Devices

n_features = len(X_train[0])
n_blocks = 3
n_qubits = n_features * n_blocks
# select backend
# for floq you'll need to create a file "floq_key" with the key in it in the current dir
# make sure it is excluded with git ignore
have_floq_key = False
if have_floq_key:
    print("You rock")
    f = open("floq_key", "r")
    floq_key = f.read().replace("\n", "")
    from remote_cirq import RemoteSimulator
    import cirq
    import sympy
    import remote_cirq
    sim = remote_cirq.RemoteSimulator(floq_key)
    dev_kernel = qml.device("cirq.simulator",
                 wires=n_qubits,
                 simulator=sim,
                 analytic=False)
else:
    print("Lame backend selected")
    dev_kernel = qml.device("default.qubit.tf", wires=n_qubits)

# # Utilities




# # Ansätze/Templates

# +
@qml.template
def angle_embedding(x, params):
    AngleEmbedding(x, wires=range(n_qubits))

@qml.template
def simple_variational_ansatz(x, params):
    qml.Hadamard(wires=[0])
    qml.CRX(params[1],wires=[0,1])
    AngleEmbedding(x, wires=range(n_qubits))
    
@qml.template
def rz_template(x, wires, param):
    qml.RZ(x, wires=wires)
    
@qml.template
def rxrzrx_template(x, wires, param):
    qml.RX(param[0], wires=wires)
    qml.RZ(x, wires=wires)
    qml.RX(param[1], wires=wires)
    
@qml.template
def ryrzry_template(x, wires, param):
    qml.RY(param[0], wires=wires)
    qml.RZ(x, wires=wires)
    qml.RY(param[1], wires=wires)
    
# This can be made into a trainable embedding that inherits from the PL embedding class later on
@qml.template
def product_embedding(x, param, wires, rotation_template=rz_template):
    m = len(wires)
    for i in range(m):
        qml.Hadamard(wires=wires[i])
        rotation_template(x[i], [wires[i]], param)
    for i in range(1, m):
        for j in range(i):
            qml.CNOT(wires=[wires[j], wires[i]])
            rotation_template((np.pi-x[i])*(np.pi-x[j]), [wires[i]], param)
#             rotation_template(x[i]*x[j], [i], param)
            qml.CNOT(wires=[wires[j], wires[i]])
    
@qml.template
def reembed(x, param, embedding, n_layers=2, n_qubits=2, n_features=2, **kwargs):
    n_block = n_qubits//n_features
    for j in range(n_layers):
        for i in range(n_block):
            embedding(x, param, wires=list(range(n_features*i, n_features*(i+1))), **kwargs)
        if j!=n_layers-1:
            for i in range(1, n_block): # Entangle neighbouring qubits that are not in the same block
                qml.CNOT(wires=[n_features*i-1, n_features*i])
        




# -

# # Kernel optimization


# +
# # Kernel optimization function
# def optimize_kernel_param(
#     kernel,
#     X_train,
#     y_train,
#     init_kernel_args,
#     samples=None,
#     seed=None,
#     optimizer=qml.AdamOptimizer,
#     optimizer_kwargs={'stepsize':0.2},
#     N_epoch=20,
#     verbose=5,
#     dx=1e-6,
#     atol=1e-3,
# ):
#     opt = optimizer(**optimizer_kwargs)
#     param = np.copy(init_kernel_args)

        
#     last_cost = 1e10
#     for i in range(N_epoch):
#         x, y = sample_data(X_train, y_train, samples, seed)
#     #     print(x, y)
#         cost_fn = lambda param: -kernel.target_alignment(x, y, param)
#         grad_fn = lambda param: (
#             -target_alignment_grad(x, y, kernel, kernel_args=param, dx=dx, assume_normalized_kernel=True),
#         )
#         param, current_cost = opt.step_and_cost(cost_fn, param, grad_fn=grad_fn)
#         if i%verbose==0:
#             print(f"At iteration {i} the polarization is {-current_cost} (params={param})")
#         if current_cost<last_cost:
#             opt_param = param.copy()
#         if np.abs(last_cost-current_cost)<atol:
#             break
#         last_cost = current_cost
        
#     return opt_param, -current_cost
# -

# This cell demonstrates that not too many samples are required to reproduce the polarization kind of okay-ish.
# This is useful because the computational cost for the polarization scale quadratically in the number of samples
# %matplotlib notebook
if False:
    P = []
    samples_ = [5*i for i in range(1, 15)]+[None]
    # samples_ = samples_[:9]
    for samples in samples_:
        print(samples, end='   ')
        x, y = sample_data(X_train, y_train, samples=samples)
        P.append(kernel.target_alignment(x, y, ()))
    print()
    sns.lineplot(x=samples_, y=P);

# # Train and validate






# +
num_param = 2
init_param = np.random.random(num_param) * 2 * np.pi
ansatz = lambda x, param: reembed(x,
                                  param,
                                  product_embedding,
                                  n_layers=2,
                                  num_wires=n_qubits, 
                                  rotation_template=rxrzrx_template)

trainable_kernel = kern.EmbeddingKernel(ansatz, dev_kernel) # WHOOP WHOOP 

# seed = 42
# samples = 10

vanilla_ta = trainable_kernel.target_alignment(X_train, y_train, np.zeros_like(init_param))
print(f"At param=[0....] the polarization is {vanilla_ta}")
start = time.time()
opt_param, last_cost = tk_lib.optimize_kernel_param(
    trainable_kernel,
    X_train, 
    y_train,
    init_kernel_args=init_param,
#     samples=samples,
#     seed=seed,
    verbose=1,
    N_epoch=20,
)
print(opt_param)
end = time.time()
print("time elapsed:", end-start)

# +
# Compare the original ansatz to a random-parameter to a polarization-trained-parameter kernel - It seems to work!
x, y = sample_data(X_train, y_train, samples=10)
svm = train_svm(trainable_kernel, x, y, np.zeros_like(init_param))
perf_train = validate(svm, x, y)
perf_test = validate(svm, X_test, y_test)
print(f"At zero parameters, the kernel has training set performance {perf_train} and test set performance {perf_test}.")
# print(f"Init parameters: {init_param}")

svm = train_svm(trainable_kernel, x, y, init_param)
perf_train = validate(svm, x, y)
perf_test = validate(svm, X_test, y_test)
print(f"At init parameters, the kernel has training set performance {perf_train} and test set performance {perf_test}.")
print(f"Init parameters: {init_param}")

svm = train_svm(trainable_kernel, x, y, opt_param)
perf_train = validate(svm, x, y)
perf_test = validate(svm, X_test, y_test)
print(f"At 'optimal' parameters, the kernel has training set performance {perf_train} and test set performance {perf_test}.")
print(f"'Optimal' parameters: {opt_param}")
# -
print("we have run a total of", dev_kernel.num_executions, "circuit executions")



svm = train_svm(trainable_kernel, X_train, y_train, np.random.random(2))
perf_train = validate(svm, X_train, y_train)
perf_test = validate(svm, X_test, y_test)
print(f"At init parameters, the kernel has training set performance {perf_train} and test set performance {' '}.")

# +
n_alpha = 15
n_beta = n_alpha
alphas = np.linspace(-np.pi/2, np.pi/2, n_alpha)
betas = np.linspace(-np.pi/2, np.pi/2, n_beta)
target_alignment = np.zeros((n_alpha,n_beta))
classification = np.zeros((n_alpha,n_beta))
# n_cpu = 1

def get_alignment_and_classification(alphas):
    ta = np.zeros((len(alphas),len(betas)))
    clsf = np.zeros((len(alphas),len(betas)))
    for i, a in enumerate(alphas):
        print(f"{i}", end='   ')
        for j, b in enumerate(betas):
            par = np.array([a,b])
            ta[i, j] = trainable_kernel.target_alignment(X_train, y_train, par)
            svm = train_svm(trainable_kernel, X_train, y_train, par)
            clsf[i, j] = validate(svm, X_train, y_train)
    return ta, clsf
            
# n_sub_alpha = n_alpha//n_cpu
# sub_alpha = [alphas[i*n_sub_alpha:(i+1)*n_sub_alpha] for i in range(n_cpu-1)] + [alphas[(n_cpu-1)*n_sub_alpha:]]
# with mproc.Pool(n_cpu) as pool:
#     TAs, CLSFs = pool.map(get_alignment_and_classification, sub_alpha)
ta, clsf = get_alignment_and_classification(alphas)
alphas, betas = np.meshgrid(alphas, betas)        

# +
# %matplotlib notebook
fig = plt.figure()
ax = fig.gca(projection='3d')

pl = ax.plot_surface(alphas, betas, target_alignment, antialiased=False, cmap=cm.coolwarm)

# +
# %matplotlib notebook
fig = plt.figure()
ax = fig.gca(projection='3d')

pl = ax.plot_surface(alphas, betas, classification, antialiased=False, cmap=cm.coolwarm)
# -

print("we have run a total of", dev_kernel.num_executions, "circuit executions")

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)





# # Evaluate all datasets



# Config field
n_blocks = 4
n_features = 3
n_param = 2
n_layers = 2
learning_rate = 0.2
use_manual_grad = True

# +
n_qubits = n_blocks * n_features

dev = qml.device("default.qubit.tf", wires=n_qubits)
ansatz = lambda x, param: reembed(x,
                                  param,
                                  product_embedding,
                                  n_layers=n_layers,
                                  n_qubits=n_qubits, 
                                  n_features=n_features,
                                  rotation_template=rxrzrx_template)
kernel = kern.EmbeddingKernel(ansatz, dev) # WHOOP WHOOP 
opt_kwargs = {'stepsize': learning_rate}

performance = []

# inds = range(9)
# inds = range(17)
inds = [1,4,6,7,10,11,12,15,16]
n_datasets = len(inds)
for dataset_index in inds:
#     X, y = tk_lib.load_data('../plots_and_data/train.txt', dataset_index) 
    X, y = tk_lib.gen_cubic_data(dataset_index)
    init_param = np.random.random(n_param) * 2 * np.pi 
#     print(init_param)
    opt_param, opt_cost = tk_lib.optimize_kernel_param(
        kernel,
        X,
        y,
        init_param=init_param,
        optimizer_kwargs=opt_kwargs,
        use_manual_grad=use_manual_grad,
    )
    
    zero_perf = tk_lib.validate(tk_lib.train_svm(kernel, X, y, np.zeros(n_param)), X, y)
    init_perf = tk_lib.validate(tk_lib.train_svm(kernel, X, y, init_param), X, y)
    opt_perf = tk_lib.validate(tk_lib.train_svm(kernel, X, y, opt_param), X, y)
    new_perf = [zero_perf, init_perf, opt_perf]
    performance.append(new_perf)
    print(new_perf)

# -

n_qubits

 # %matplotlib notebook
indices = np.array(list(range(n_datasets)))
plt.scatter(indices-0.1, np.array(performance)[:, 0], color='r', marker='x', label='Zero')
plt.scatter(indices, np.array(performance)[:, 1], color='b', marker='o', label='Init')
plt.scatter(indices+0.1, np.array(performance)[:, 2], color='g', marker='d', label='Opt')
for ind in indices[:-1]:
    plt.plot([ind+0.5]*2, [np.min(performance), np.max(performance)], color='0.7', ls=':')
plt.legend()
plt.xticks(indices)
plt.xlabel('Dataset')
plt.ylabel('Training set performance')






























