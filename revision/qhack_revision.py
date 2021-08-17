# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from pennylane import numpy as np
import pennylane as qml
import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition

import matplotlib.pyplot as plt


np.random.seed(42)


# -

# ## Data generation functions

def create_data():
    # Load data
    data = pd.read_csv("datasets/bank.csv", sep=";")

    #print(data)
    #print(data[4500:])

    # clean data
    for column_index in range(len(data.columns)):
        print("column", column_index)
        k = data.keys()
        #data[k[-1]]
        unique_entries = data[k[column_index]].unique()
        print("unique_entries", unique_entries)
        for entry_index in range(len(unique_entries)):
            unique_entry = unique_entries[entry_index]
            if str(unique_entry).lstrip('-').isnumeric() == False:
                print(unique_entry, entry_index)
                data[k[column_index]] = data[k[column_index]].replace(to_replace=unique_entry, value=float(entry_index))
    #print(data[4500:])

    # define data
    data = data.astype(float)
    #print(data)

    # Normalize data
    scaler = MinMaxScaler()
    data = pd.DataFrame(data=scaler.fit_transform(data), columns=data.columns)
    #print(data[4500:])

    X = data.iloc[:, 0:len(data.columns)-1].to_numpy()
    y = data.iloc[:, len(data.columns)-1].to_numpy()
    
    return X, y


def prep_data(X, y, feature_dim):
    pca = decomposition.PCA(n_components=feature_dim)
    X = X[:, 0:feature_dim]
    pca.fit(X)
    X_trans = pca.transform(X)
    return X_trans, y



# ## Plot function





# ## Other helper functions

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)


def plot_decision_boundaries(classifier, ax, N_gridpoints=14):
    _xx, _yy = np.meshgrid(np.linspace(-1, 1, N_gridpoints), np.linspace(-1, 1, N_gridpoints))

    _zz = np.zeros_like(_xx)
    for idx in np.ndindex(*_xx.shape):
        _zz[idx] = classifier.predict(np.array([_xx[idx], _yy[idx]])[np.newaxis, :])

    plot_data = {"_xx": _xx, "_yy": _yy, "_zz": _zz}
    ax.contourf(
        _xx,
        _yy,
        _zz,
        cmap=mpl.colors.ListedColormap(["#FF0000", "#0000FF"]),
        alpha=0.2,
        levels=[-1, 0, 1],
    )
    plot_double_cake_data(X, Y, ax)

    return plot_data



# ## Init and visualize



# +
X, Y = create_data()

# randomize
randomize = np.arange(len(X))
np.random.shuffle(randomize)
X = X[randomize]
Y = Y[randomize]

#limit
#cut_off = 10
#X_cut = X[:cut_off]
#y_cut = Y[:cut_off]

feature_dim = 5
X, Y = prep_data(X, Y, feature_dim)
cut_off = 20
X = X[0:cut_off]
Y = Y[0:cut_off]
X.shape

# -



# ## Ansatz preparation functions

# +
def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])
    
def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires))    


# -

# The kernel function itself is now obtained by looking at the probability 
# of observing the all-zero state at the end of the kernel circuit – 
# because of the ordering in qml.probs, this is the first entry:
def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]



# ## Init network

# +
dev = qml.device("default.qubit", wires=5, shots=None)
wires = dev.wires.tolist()

@qml.qnode(dev)
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)
# -



# +
# init parameters
init_params = random_params(num_wires=5, num_layers=6)

# calculate kernel matrix
init_kernel = lambda x1, x2: kernel(x1, x2, init_params)
K_init = qml.kernels.square_kernel_matrix(X, init_kernel, assume_normalized_kernel=True)

with np.printoptions(precision=3, suppress=True):
    print(K_init)
# -

# ## Train an SVM

from sklearn.svm import SVC

# +
# train alpha and beta
svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(X, Y)

# check the performance
print(f"The accuracy of the kernel with random parameters is {accuracy(svm, X, Y):.3f}")
#init_plot_data = plot_decision_boundaries(svm, plt.gca())
# -



# ## Train the QEK

# init and evaluate QEK
kta_init = qml.kernels.target_alignment(X, Y, init_kernel, assume_normalized_kernel=True)
print(f"The kernel-target alignment for our dataset and random parameters is {kta_init:.3f}")


def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """Kernel-target alignment between kernel and labels."""

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    print("K", K)
    print("norm", norm)
    inner_product = inner_product / norm

    return inner_product


Y

# +
params = init_params
opt = qml.GradientDescentOptimizer(0.2)

for i in range(500):
    # Choose subset of datapoints to compute the KTA on.
    subset = np.random.choice(list(range(len(X))), 4)
    print(subset)
    # Define the cost function for optimization
    cost = lambda _params: -target_alignment(
        X[subset],
        Y[subset],
        lambda x1, x2: kernel(x1, x2, _params),
        assume_normalized_kernel=True,
    )
    # Optimization step
    params = opt.step(cost, params)

    # Report the alignment on the full dataset every 50 steps.
    if (i + 1) % 50 == 0:
        current_alignment = target_alignment(
            X,
            Y,
            lambda x1, x2: kernel(x1, x2, params),
            assume_normalized_kernel=True,
        )
        print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

# +
# First create a kernel with the trained parameters baked into it.
trained_kernel = lambda x1, x2: kernel(x1, x2, params)

# Second create a kernel matrix function using the trained kernel.
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)

# Note that SVC expects the kernel argument to be a kernel matrix function.
svm_trained = SVC(kernel=trained_kernel_matrix).fit(X, Y)
# -

accuracy_trained = accuracy(svm_trained, X, Y)
print(f"The accuracy of a kernel with trained parameters is {accuracy_trained:.3f}")




































