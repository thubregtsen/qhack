# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
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
import time

np.random.seed(42)
# -


# # Data


data = 'cake'
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

# # Devices

# +
n_qubits = len(X_train[0]) # -> equals number of features

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


# -

# # Utilities

# +

def sample_data(X_train, Y_train, samples=None, seed=None):
    m = len(Y_train)
    if samples is None or samples>m:
        x = X_train
        y = Y_train
        samples = m
    else:
        if seed is None:
            seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        sampled = np.random.choice(list(range(m)), samples)
        x = X_train[sampled]
        y = Y_train[sampled]
    return x, y

def target_alignment_grad(X_train, y_train, kernel, kernel_args, dx=1e-6, **kwargs):
    g = np.zeros_like(kernel_args)
    shifts = np.eye(len(kernel_args))*dx/2
    for i, shift in enumerate(shifts):
        ta_plus = kernel.target_alignment(X_train, y_train, kernel_args+shift)
        ta_minus = kernel.target_alignment(X_train, y_train, kernel_args-shift)
        g[i] = (ta_plus-ta_minus)/dx
#     print(g)
    return g


# -


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
def product_embedding(x, param, rotation_template=rz_template):
    m = len(x)
    for i in range(m):
        qml.Hadamard(wires=[i])
        rotation_template(x[i], [i], param)
    for i in range(1, m):
        for j in range(i):
            qml.CNOT(wires=[j, i])
            rotation_template((np.pi-x[i])*(np.pi-x[j]), [i], param)
#             rotation_template(x[i]*x[j], [i], param)
            qml.CNOT(wires=[j, i])


# -

# # Kernel optimization


# Kernel optimization function
def optimize_kernel_param(
    kernel,
    X_train,
    y_train,
    init_kernel_args,
    samples=None,
    seed=None,
    normalize=False,
    optimizer=qml.AdamOptimizer,
    optimizer_kwargs={'stepsize':0.2},
    N_epoch=20,
    verbose=5,
    dx=1e-6,
    atol=1e-3,
):
    opt = optimizer(**optimizer_kwargs)
    param = np.copy(init_kernel_args)

        
    last_cost = 1e10
    for i in range(N_epoch):
        x, y = sample_data(X_train, y_train, samples, seed)
    #     print(x, y)
        cost_fn = lambda param: -kernel.target_alignment(x, y, param)
        grad_fn = lambda param: (-target_alignment_grad(
            x, y, kernel, kernel_args=param, dx=dx, assume_normalized_kernel=True
        ),)
        param, current_cost = opt.step_and_cost(cost_fn, param)#, grad_fn=grad_fn)
        if i%verbose==0:
            print(f"At iteration {i} the polarization is {-current_cost} (params={param})")
        if np.abs(last_cost-current_cost)<atol:
            break
        last_cost = current_cost
        
    return param, -current_cost

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
#         P.append(polarization(kernel, X_train, y_train, samples=samples, normalize=True))
    print()
    sns.lineplot(x=samples_, y=P);


# # Train and validate

# +
def train_svm(kernel, X_train, y_train, param):
    def kernel_matrix(A, B):
        """Compute the matrix whose entries are the kernel
           evaluated on pairwise data from sets A and B."""
        if A.shape==B.shape and np.allclose(A, B):
            return kernel.square_kernel_matrix(A, param)
        
        return kernel.kernel_matrix(A, B, param)
    
    svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)
    return svm
    
def validate(model, X, y_true):
    y_pred = model.predict(X)
    errors = np.sum(np.abs((y_true - y_pred)/2))
    return (len(y_true)-errors)/len(y_true)

# -



# +
num_param = 2
init_par = np.random.random(num_param) * 2 * np.pi
ansatz = lambda x, param: product_embedding(x, param, rxrzrx_template)

trainable_kernel = kern.EmbeddingKernel(ansatz, dev_kernel) # WHOOP WHOOP 

# seed = 42
samples = 10

normalize = True

vanilla_ta = trainable_kernel.target_alignment(X_train, y_train, np.zeros_like(init_par))
print(f"At param=[0....] the polarization is {vanilla_ta}")
start = time.time()
opt_param, last_cost = optimize_kernel_param(
    trainable_kernel,
    X_train, 
    y_train,
    init_kernel_args=init_par,
    samples=samples,
#     seed=seed,
    normalize=normalize,
    verbose=1,
    N_epoch=20,
)
end = time.time()
print("time elapsed:", end-start)

# +
# Compare the original ansatz to a random-parameter to a polarization-trained-parameter kernel - It seems to work!
x, y = sample_data(X_train, y_train, samples=10)
svm = train_svm(trainable_kernel, x, y, np.zeros_like(init_par))
perf_train = validate(svm, x, y)
perf_test = validate(svm, X_test, y_test)
print(f"At zero parameters, the kernel has training set performance {perf_train} and test set performance {perf_test}.")

svm = train_svm(trainable_kernel, x, y, init_par)
perf_train = validate(svm, x, y)
perf_test = validate(svm, X_test, y_test)
print(f"At init parameters, the kernel has training set performance {perf_train} and test set performance {perf_test}.")

svm = train_svm(trainable_kernel, x, y, opt_param)
perf_train = validate(svm, x, y)
perf_test = validate(svm, X_test, y_test)
print(f"At 'optimal' parameters, the kernel has training set performance {perf_train} and test set performance {perf_test}.")
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




















































