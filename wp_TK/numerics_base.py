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

# **To be able to run this notebook you need to install the modified PennyLane version that contains the `qml.kernels` module via**
# ```
# pip install git+https://www.github.com/johannesjmeyer/pennylane@kernel_module --upgrade
# ```

#

#

# +
import pennylane as qml
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC

np.random.seed(42)
# -

#



#



#

# +
def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding Ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])
        
    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

@qml.template
def ansatz(x, params, wires):
    """The embedding Ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))
        
def random_params(num_wires, num_layers):
    return np.random.uniform(0, 2*np.pi, (num_layers, 2, num_wires))


# -

#

dev = qml.device("default.qubit", wires=5)
wires = list(range(5))



#



#



#



#



#



#

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)



#



#



#

#



#



#





















# +
# create dataset
samples = 10
features = 2
## choose random input
X = np.random.random((2*samples,features))
## choose random params for the kernel, which will be our ideal parameters
ideal_params = random_params(5, 6)
# init the embedding kernel
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x, params, wires), dev)

#generate train data
X_train = X[:samples]
## choose random y targets in a balanced way
indices = np.arange(samples)
np.random.shuffle(indices)
indices = indices[:int(samples/2)] # uneven will get rounded
y_init = np.ones((10))
y_init[indices] = -1
## fit the SVM
svm_init = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, ideal_params)).fit(X_train, y_init)
## and determine what the SVM can actually predict with our ideal parameters
y_train = svm_init.predict(X_train)

#generate train data
X_test = X[samples:]
## choose random y targets in a balanced way
indices = np.arange(samples)
np.random.shuffle(indices)
indices = indices[:int(samples/2)] # uneven will get rounded
y_init = np.ones((10))
y_init[indices] = -1
## fit the SVM
svm_init = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, ideal_params)).fit(X_test, y_init)
## and determine what the SVM can actually predict with our ideal parameters
y_test = svm_init.predict(X_train)

print("X_train:", X_train)
print("y_train:", y_train)
print("X_test:", X_test)
print("y_test:", y_test)

# -

# evaluate the performance with random parameters for the kernel
## choose random params for the kernel
params = random_params(5, 6)
## fit the SVM
svm_untrained_kernel = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, params)).fit(X_train, y_train)
untrained_accuracy = accuracy(svm_untrained_kernel, X_test, y_test)
print("without kernel training accuracy", untrained_accuracy)

# evaluate the performance with trained parameters for the kernel
## train the kernel
opt = qml.GradientDescentOptimizer(2.5)
for i in range(500):
    subset = np.random.choice(list(range(len(X_train))), 4)
    params = opt.step(lambda _params: -k.target_alignment(X_train[subset], y_train[subset], _params), params)
    
    if (i+1) % 50 == 0:
        print("Step {} - Alignment on train = {:.3f}".format(i+1, k.target_alignment(X_train, y_train, params)))

## fit the SVM
svm_trained_kernel = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, params)).fit(X_train, y_train)
trained_accuracy = accuracy(svm_untrained_kernel, X_test, y_test)
print("with kernel training accuracy", trained_accuracy)

## fit the SVM
svm_ideal_kernel = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, ideal_params)).fit(X_train, y_train)
ideal_accuracy = accuracy(svm_ideal_kernel, X_train, y_train)
print("ideal accuracy", trained_accuracy)




