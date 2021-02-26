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

# # Quantum Embedding Kernels for MNIST with floq
#
# _Authors: Peter-Jan Derks, Paul Fährmann, Elies Gil-Fuster, Tom Hubregtsen, Johannes Jakob Meyer and David Wierichs_

# ## MNIST
#
# In this demonstration, we will have a look at the popular MNIST dataset, consisting of tens of thousands of $28 \times 28$ pixel images. To make this tractable for simulation we will only work with a small subset of MNIST here.

# +
import pennylane as qml
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(2658)
# -

from keras.datasets import mnist

# Let's now have a look at our dataset. In our example, we will work with 6 sectors:

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Let us now extract and rescale our training data

# +
print(train_X.shape)

train_idx0 = np.argwhere(train_y == 0)[:10]
train_X0 = train_X[train_idx0].squeeze() * np.pi/255

train_idx1 = np.argwhere(train_y == 1)[:10]
train_X1 = train_X[train_idx1].squeeze() * np.pi/255

# +
gs = mpl.gridspec.GridSpec(2, 10) 
fig = plt.figure(figsize=(16,4))

#Using the 1st row and 1st column for plotting heatmap

for j in range(10):
    ax=plt.subplot(gs[0, j])
    plt.imshow(train_X0[j], cmap=plt.get_cmap('gray'))
    ax.axis("off")
    
    ax=plt.subplot(gs[1, j])
    plt.imshow(train_X1[j], cmap=plt.get_cmap('gray'))
    ax.axis("off")    
# -

X = np.vstack([train_X0, train_X1])
y = np.hstack([[-1]*10, [1]*10])


# Next step: rescaling

# ## Defining a Quantum Embedding Kernel
#
# PennyLane's `kernels` module allows for a particularly simple implementation of Quantum Embedding Kernels. The first ingredient we need for this is an _ansatz_ that represents the unitary $U(\boldsymbol{x})$ we use for embedding the data into a quantum state. We will use a structure where a single layer is repeated multiple times:

# +
def layer(x, params, wires, i0=0, inc=1):
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])
        
    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

@qml.template
def ansatz(x, params, wires):
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))
        
def random_params(num_wires, num_layers):
    return np.random.uniform(0, 2*np.pi, (num_layers, 2, num_wires))


# -

# We are now in a place where we can create the embedding. Together with the ansatz we only need a device to run the quantum circuit on. For the purposes of this tutorial we will use PennyLane's `default.qubit` device with 5 wires.

dev = qml.device("lightning.qubit", wires=3)
wires = list(range(3))
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x.flatten(), params, wires), dev)

# And this was all of the magic! The `EmbeddingKernel` class took care of providing us with a circuit that calculates the overlap. Before we can take a look at the kernel values we have to provide values for the variational parameters. We will initialize them such that the ansatz circuit has $6$ layers.

init_params = random_params(3, 8)

# Now we can have a look at the kernel value between the first and the second datapoint:

print("The kernel value between the first and second datapoint is {:.3f}".format(k(train_X0[0], train_X1[0], init_params)))

# The mutual kernel values between all elements of the dataset form the _kernel matrix_. We can inspect it via the `square_kernel_matrix` method:

# +
K_init = k.square_kernel_matrix(X, init_params)

with np.printoptions(precision=3, suppress=True):
    print(K_init)
# -

# ## Using the Quantum Embedding Kernel for predictions
#
# The quantum kernel alone can not be used to make predictions on a dataset, becaues it essentially just a tool to measure the similarity between two datapoints. To perform an actual prediction we will make use of scikit-learns support vector classifier (SVC). 

from sklearn.svm import SVC

# The `SVC` class expects a function that maps two sets of datapoints to the corresponding kernel matrix. This is provided by the `kernel_matrix` property of the `EmbeddingKernel` class, we only have to use a lambda construction to include our parameters. Once we provide this, we can fit the SVM on our Quantum Embedding Kernel circuit. Note that this does not train the parameters in our circuit. 

svm = SVC(kernel="precomputed").fit(K_init, y)

# To see how well our classifier performs we will measure what percentage it classifies correctly.

print("The accuracy of a kernel with random parameters is {:.3f}".format(
    1 - np.count_nonzero(svm.predict(K_init) - y) / len(y))
)

# +
test_idx0 = np.argwhere(train_y == 0)[10:20]
test_X0 = train_X[train_idx0].squeeze() * np.pi/255

test_idx1 = np.argwhere(train_y == 1)[10:20]
test_X1 = train_X[train_idx1].squeeze() * np.pi/255

X_test = np.vstack([test_X0, test_X1])
y_test = np.hstack([[-1]*10, [1]*10])
# -

K_pred = k.kernel_matrix(X_test, X, init_params)

print("The test accuracy of a kernel with random parameters is {:.3f}".format(
    1 - np.count_nonzero(svm.predict(K_pred) - y_test) / len(y_test))
)


