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
import torch
from torch.nn.functional import relu

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor
import pennylane.kernels as kern

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import jupytext

import time

np.random.seed(42)
# -


# # Data


# +
# load the data
X, y = load_iris(return_X_y=True) 

print("The dataset contains X and y, each of length", len(X))
print("X contains", len(X[0]), "features")
print("y contains the following classes", np.unique(y))

# pick inputs and labels from the first two classes only,
# corresponding to the first 100 samples
# -> meanig y now consists of 2 classes: 0, 1; still stored in order, balanced 50:50
X = X[:100,:2]
y = y[:100]

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
# -

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
    print("Lame backed selected")
    dev_kernel = qml.device("default.qubit", wires=n_qubits)


# -

# David, do you use this? Please either remove the comment or the code
projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1



# # Kernel


# +
def polarization(kernel, X_train, Y_train, kernel_args=(), samples=None, seed=None, normalize=False):
    """Compute the polarization of a given kernel on training data.
    Args:
      kernel (qml.kernels.Kernel): The (variational) quantum kernel (imaginary class that does not exist yet)
      X_train (ndarray): Training data inputs.
      Y_train (ndarray): Training data outputs.
      samples (int): Number of samples to draw from the training data. If None, all data will be used.
      seed (int): Seed for random sampling from the data, if None, a random seed will be used.
    Returns:
      P (float): The polarization of the kernel on the given data.
    """        
#     print(kernel_args, samples, seed, normalize)
    if seed is None:
        seed = np.random.randint(0, 1000000)
    m = len(Y_train)
    seed = 42 # NOTE: I FIXED THE SEED with a guarfor performance testing
    np.random.seed(seed)
    if samples is None or samples>m:
        x = X_train
        y = Y_train
        samples = m
    else:
        sampled = np.random.choice(list(range(m)), samples)
        x = X_train[sampled]
        y = Y_train[sampled]
    
    P = 0
    K = 0
    # Only need to compute the upper right triangle of the kernel matrix and y_correl_matrix (they are symmetric)
    # Actually, the diagonal is usually going to be 1 (for y_correl it is for labels +-1), but we can see that later
    for i, (x1, y1) in enumerate(zip(x, y)):
        k = kernel(x1, x1, kernel_args)
        K += np.abs(k)**2
        P += y1*y1 * k # Usually will be 1
        for x2, y2 in zip(x[i+1:], y[i+1:]):
            k = kernel(x1, x2, kernel_args)
            K += 2* np.abs(k)**2
            P += 2 * y1 * y2 * k
    
    if normalize:
        P /= np.sqrt(K) * len(y)
    else:
        # Adapt to sampling number if we do not normalize anyways. (When normalizing, this would cancel for K and P)
        P /= (samples/m)**2
            
    return P

def polarization_grad(kernel, X_train, y_train, kernel_args, dx=1e-6, **kwargs):
    g = np.zeros_like(kernel_args)
    shifts = np.eye(len(kernel_args))*dx/2
    need_to_set_seed = 'seed' not in kwargs
    for i, shift in enumerate(shifts):
        # Even if no seed is given, fix the seed per gradient entry
        if need_to_set_seed:
            kwargs['seed'] = np.random.randint(0,100000) 
        upper = polarization(kernel, X_train, y_train, kernel_args=kernel_args+shift, **kwargs)
        lower = polarization(kernel, X_train, y_train, kernel_args=kernel_args-shift, **kwargs)
        g[i] = (upper-lower)/dx
#     print(g)
    return g


# -


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
    cost_fn = lambda param: -polarization(kernel, X_train, y_train, kernel_args=param, samples=samples,
                                          seed=seed, normalize=normalize)
    grad_fn = lambda param: (-polarization_grad(
        kernel, X_train, y_train, kernel_args=param, samples=samples, seed=seed, dx=dx, normalize=normalize,
    ),)
#     def grad_fn(param):
#         g = qml.grad(cost_fn)(param)
#         print(g)
#         return g
        
    last_cost = 1e10
    for i in range(N_epoch):
        param, current_cost = opt.step_and_cost(cost_fn, param, grad_fn=grad_fn)
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
        P.append(polarization(kernel, X_train, y_train, samples=samples, normalize=True))
    print()
    sns.lineplot(x=samples_, y=P);


# # Train and validate

# +
def train_svm(kernel, X_train, y_train, param, assume_normalized_kernel=True):
    def kernel_matrix(A, B):
        M = np.eye(len(A))
        for i, a in enumerate(A):
            for j, b in enumerate(B[i+1:]):
                M[i, j] = kernel(a, b, param)
                M[j, i] = M[i, j]
        return M

#     kernel_matrix = lambda A, B: np.array([[kernel(a, b, param) for b in B] for a in A])
    svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)
    return svm
    
def validate(model, X, y_true):
    y_pred = model.predict(X)
    errors = np.sum(np.abs([(y_true[i] - y_pred[i])/2 for i in range(len(y_true))]))
    return (len(y_true)-errors)/len(y_true)



# +
@qml.template
def ansatz(x, params):
    qml.Hadamard(wires=[0])
    qml.CRX(params[1],wires=[0,1])
    AngleEmbedding(x, wires=range(n_qubits))

trainable_kernel = kern.EmbeddingKernel(ansatz, dev_kernel) # WHOOP WHOOP 

init_par = np.random.random(3)
seed = 42
samples = 30
normalize = True
vanilla_polarization = polarization(trainable_kernel, X_train, y_train, kernel_args=np.zeros_like(init_par),
                                    samples=samples, seed=seed, normalize=normalize,)
print(f"At param=[0....] the polarization is {vanilla_polarization}")
start = time.time()
opt_param, last_cost = optimize_kernel_param(
    trainable_kernel,
    X_train, 
    y_train,
    init_kernel_args=init_par,
    samples=samples,
    seed=seed,
    normalize=normalize,
    verbose=1,
)
end = time.time()
print("time elapsed:", end-start)

# +
# Compare the original ansatz to a random-parameter to a polarization-trained-parameter kernel - It seems to work!
svm = train_svm(trainable_kernel, X_train, y_train, np.zeros_like(init_par))
perf_train = validate(svm, X_train, y_train)
perf_test = validate(svm, X_test, y_test)
print(f"At zero parameters, the kernel has training set performance {perf_train} and test set performance {perf_test}.")

svm = train_svm(trainable_kernel, X_train, y_train, init_par)
perf_train = validate(svm, X_train, y_train)
perf_test = validate(svm, X_test, y_test)
print(f"At init parameters, the kernel has training set performance {perf_train} and test set performance {perf_test}.")

svm = train_svm(trainable_kernel, X_train, y_train, opt_param)
perf_train = validate(svm, X_train, y_train)
perf_test = validate(svm, X_test, y_test)
print(f"At 'optimal' parameters, the kernel has training set performance {perf_train} and test set performance {perf_test}.")
# -
print("we have run a total of", dev_kernel.num_executions, "circuit executions")


































































