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
import time

import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error



printing = False

# downsize our data         
cut_off = 5
# define the number of desired dimensions
feature_dim = 12 # min dim 12 for bank dataset
optimization_iterations = 50
optimization_samples = 4
seed = 43

import sys
# -

if "ipykernel" not in sys.argv[0]:
    # running in cmd
    cut_off = int(sys.argv[1])
    optimization_iterations = int(sys.argv[2])
    optimization_samples = int(sys.argv[3])
    seed = int(sys.argv[4])
    print("cut_off set", cut_off, "optimization_iterations", optimization_iterations, "optimization_samples", optimization_samples, "seed", seed)
np.random.seed(seed)

start = time.time()


# ## Data generation functions

def create_data():
    # Load data
    data = pd.read_csv("datasets/bank.csv", sep=";")

    #print(data)
    #print(data[4500:])

    # clean data
    for column_index in range(len(data.columns)):
        if printing == True:
            print("column", column_index)
        k = data.keys()
        #data[k[-1]]
        unique_entries = data[k[column_index]].unique()
        if printing == True:
            print("unique_entries", unique_entries)
        for entry_index in range(len(unique_entries)):
            unique_entry = unique_entries[entry_index]
            if str(unique_entry).lstrip('-').isnumeric() == False:
                if printing == True:
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
    y = (y*2) -1 # [0,1] -> [0,2] -> [-1,1]
    
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


def acc(a,b):
    total_correct = 0
    for i in range(len(a)):
        if a[i]*b[i] >= 0:
            total_correct += 1
        #else:
        #    print("incorrect")
    return (len(a)-total_correct)/len(a)

# ## Init and visualize



# +
# take the dataset, and parse it do that all data is in R
X, Y = create_data()
# PCA
X, Y = prep_data(X, Y, feature_dim)
# check if we still have unique data after dimensionality reduction
if not (len(X) == len(np.unique(X, axis=0))):
    print(len(X), len(np.unique(X, axis=0)))
    assert False, "DATA NOT UNIQUE, DUPLICATES DETECTED!"

# make sure we have balanced data
X_neg = []
X_pos = []
Y_neg = []
Y_pos = []
for i in range(len(Y)):
    if Y[i] < 0:
        X_neg.append(X[i])
        Y_neg.append(Y[i])
    else:
        X_pos.append(X[i])
        Y_pos.append(Y[i])
X_neg = np.asarray(X_neg)
X_pos = np.asarray(X_pos)
Y_neg = np.asarray(Y_neg)
Y_pos = np.asarray(Y_pos)

# shuffle our data, positive and negative samples seperately
randomize_neg = np.arange(len(X_neg))
np.random.shuffle(randomize_neg)
X_neg = X_neg[randomize_neg]
Y_neg = Y_neg[randomize_neg]
randomize_pos = np.arange(len(X_pos))
np.random.shuffle(randomize_pos)
X_pos = X_pos[randomize_pos]
Y_pos = Y_pos[randomize_pos]


# first the stitching and reshufling of the train data
X_train = np.vstack((X_neg[0:cut_off], X_pos[0:cut_off]))
Y_train = np.hstack((Y_neg[0:cut_off], Y_pos[0:cut_off]))
randomize_all = np.arange(len(X_train))
np.random.shuffle(randomize_all)
X_train = X_train[randomize_all]
Y_train = Y_train[randomize_all]
# then the stitching and reshufling of the validation data
X_val = np.vstack((X_neg[cut_off: 2*cut_off], X_pos[cut_off: 2*cut_off]))
Y_val = np.hstack((Y_neg[cut_off: 2*cut_off], Y_pos[cut_off: 2*cut_off]))
randomize_val = np.arange(len(X_val))
np.random.shuffle(randomize_val)
X_val = X_val[randomize_val]
Y_val = Y_val[randomize_val]

if printing == True:
    print("X_train", X_train)
    print("Y_train", Y_train)
    print("X_val", X_val)
    print("Y_val", Y_val)

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



# ## Init QEK

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
K_init = qml.kernels.square_kernel_matrix(X_train, init_kernel, assume_normalized_kernel=True)

if printing == True:
    with np.printoptions(precision=3, suppress=True):
        print(K_init)
# -

# ## Train an SVM

from sklearn.svm import SVC



# +
# train alpha and beta
svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(X_train, Y_train)

# check the performance
Y_pred = svm.predict(X_train)
random_on_train = acc(Y_pred, Y_train)
random_on_train_l1 = np.round(mean_absolute_error(Y_pred, Y_train), 3)
random_on_train_l2 = np.round(mean_squared_error(Y_pred, Y_train), 3)
Y_pred = svm.predict(X_val)
random_on_val = acc(Y_pred, Y_val)
random_on_val_l1 = np.round(mean_absolute_error(Y_pred, Y_val), 3)
random_on_val_l2 = np.round(mean_squared_error(Y_pred, Y_val), 3)
if printing == True:
    print(f"Random parameter accuracy on train {random_on_train:.3f}")
    print(f"Random parameter accuracy on validate {random_on_val:.3f}")
#init_plot_data = plot_decision_boundaries(svm, plt.gca())
# -



# ## Train the QEK

# init and evaluate QEK
kta_init = qml.kernels.target_alignment(X_train, Y_train, init_kernel, assume_normalized_kernel=True)
if printing == True:
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
    #print("K", K)
    #print("norm", norm)
    inner_product = inner_product / norm

    return inner_product



# +
params = init_params
opt = qml.GradientDescentOptimizer(0.2)

start_opt = time.time()
alignment = []
for i in range(optimization_iterations):
    # Choose subset of datapoints to compute the KTA on.
    subset = np.random.choice(list(range(len(X_train))), optimization_samples)
    #print(subset)
    # Define the cost function for optimization
    cost = lambda _params: -target_alignment(
        X_train[subset],
        Y_train[subset],
        lambda x1, x2: kernel(x1, x2, _params),
        assume_normalized_kernel=True,
    )
    # Optimization step
    params = opt.step(cost, params)

    # Report the alignment on the full dataset every 50 steps.
    if (i + 1) % 500 == 0:
        current_alignment = target_alignment(
            X_train,
            Y_train,
            lambda x1, x2: kernel(x1, x2, params),
            assume_normalized_kernel=True,
        )
        alignment.append(current_alignment)
        print(f"Step {i+1} - Alignment = {current_alignment:.3f}")
        
end_opt = time.time()

# +
# First create a kernel with the trained parameters baked into it.
trained_kernel = lambda x1, x2: kernel(x1, x2, params)

# Second create a kernel matrix function using the trained kernel.
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)

# Note that SVC expects the kernel argument to be a kernel matrix function.
svm_trained = SVC(kernel=trained_kernel_matrix).fit(X_train, Y_train)
# -

Y_pred = svm_trained.predict(X_train)
opt_on_train = acc(Y_pred, Y_train)
opt_on_train_l1 = np.round(mean_absolute_error(Y_pred, Y_train), 3)
opt_on_train_l2 = np.round(mean_squared_error(Y_pred, Y_train), 3)
Y_pred = svm_trained.predict(X_val)
opt_on_val = acc(Y_pred, Y_val)
opt_on_val_l1 = np.round(mean_absolute_error(Y_pred, Y_val), 3)
opt_on_val_l2 = np.round(mean_squared_error(Y_pred, Y_val), 3)
if printing == True:
    print(f"Trained parameter accuracy on train {opt_on_train:.3f}")
    print(f"Trained parameter accuracy on train {opt_on_val:.3f}")

end = time.time()

total_time = end-start
opt_time = end_opt-start_opt
if printing == True:
    print("total time elapsed", total_time)
    print("optimization time elapsed", opt_time)

# +



kernels = ["linear", "sigmoid", "rbf"] # poly takes 50x time so excluded
loss = {}
loss["l1"]={}
loss["l2"]={}
loss["acc"]={}
for kernel in kernels:
    loss["l1"][kernel]={}
    loss["l2"][kernel]={}
    loss["acc"][kernel]={}
longest_name = np.max([len(x) for x in kernels])
if printing == True:
    print("kernel;", "l1 loss;""l2 loss;")

predictions = {}
for kernel in kernels: 
    classifier = SVR(gamma='auto', kernel=kernel)
    classifier.fit(X_train, Y_train)
    predictions[kernel] = classifier.predict(X_val)
    
for kernel in kernels:
    normalized_name = kernel + ' ' * (longest_name-len(kernel)) # for spacing
    loss["acc"][kernel][feature_dim] = np.round(acc(predictions[kernel], Y_val), 3)
    loss["l1"][kernel][feature_dim] = np.round(mean_absolute_error(predictions[kernel], Y_val), 3)
    loss["l2"][kernel][feature_dim] = np.round(mean_squared_error(predictions[kernel], Y_val), 3)
    if printing == True:
        print(feature_dim, normalized_name, loss["l1"][kernel][feature_dim], 
            loss["l2"][kernel][feature_dim])
if printing == True:
    print("") # extra newline

# -







print(";".join((str(seed), str(feature_dim), str(cut_off), str(int(opt_time)), str(int(total_time)), str(optimization_iterations), str(optimization_samples), str(random_on_train), str(random_on_train_l1), str(random_on_train_l2), str(random_on_val), str(random_on_val_l1), str(random_on_val_l2), str(opt_on_train), str(opt_on_train_l1), str(opt_on_train_l2), str(opt_on_val), str(opt_on_val_l1), str(opt_on_val_l2), str(loss["acc"]["linear"][feature_dim]), str(loss["l1"]["linear"][feature_dim]), str(loss["l2"]["linear"][feature_dim]), str(loss["acc"]["sigmoid"][feature_dim]), str(loss["l1"]["sigmoid"][feature_dim]), str(loss["l2"]["sigmoid"][feature_dim]), str(loss["acc"]["rbf"][feature_dim]), str(loss["l1"]["rbf"][feature_dim]), str(loss["l2"]["rbf"][feature_dim]))))



mean_absolute_error([1,-1], [-1,-1])



















# ## 
