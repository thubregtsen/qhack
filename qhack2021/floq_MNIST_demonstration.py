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
#
# In this demonstration we showcase how simple it is to use the `qml.kernels` module to perform classification with large datapoints using `floq` to offload the heavy computation of wide circuits.

# ## MNIST
#
# We will use the popular MNIST dataset, consisting of tens of thousands of $28 \times 28$ pixel images. To make this tractable for simulation on our hardware, we will only work with a small subset of MNIST.

# +
import pennylane as qml
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import remote_cirq

np.random.seed(2658)
# -

# For convenience, we will use `keras` for loading the dataset.

# +
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()
# -

# Let us now extract the first 10 zeros and the first 10 ones as our training data. We will also rescale the images, whos pixel values are given with values in the interval $[0, 255]$, to the interval $[0, \pi]$ to be compatible with embeddings that use angles.

# +
print(train_X.shape)

sample_size = 5

train_idx0 = np.argwhere(train_y == 0)[:sample_size]
train_X0 = train_X[train_idx0].squeeze() * np.pi/255

train_idx1 = np.argwhere(train_y == 1)[:sample_size]
train_X1 = train_X[train_idx1].squeeze() * np.pi/255
# -

# Now let us have a look at our training data:

# +
gs = mpl.gridspec.GridSpec(2, sample_size) 
fig = plt.figure(figsize=(16,4))

for j in range(sample_size):
    ax=plt.subplot(gs[0, j])
    plt.imshow(train_X0[j], cmap=plt.get_cmap('gray'))
    ax.axis("off")
    
    ax=plt.subplot(gs[1, j])
    plt.imshow(train_X1[j], cmap=plt.get_cmap('gray'))
    ax.axis("off")    
# -

# With the zeros and ones extracted, we can now create the actual variables we use for the training of our model:

X = np.vstack([train_X0, train_X1])
y = np.hstack([[-1]*sample_size, [1]*sample_size])


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

# We are now in a place where we can create the embedding. Together with the ansatz we only need a device to run the quantum circuit on. For the purposes of this tutorial we will use the `floq` device with $28$ wires. Note that we need to flatten the input data to our ansatz, as the ansatz expects a flat array but the datapoints are two dimensional images.

# +
N_WIRES = 26 # can also be 28 x 28
N_LAYERS = 31

API_KEY = "YOUR KEY"
sim = remote_cirq.RemoteSimulator(API_KEY)
dev = qml.device("cirq.simulator",
                 wires=N_WIRES,
                 simulator=sim,
                 analytic=False)

wires = list(range(N_WIRES))
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x.flatten(), params, wires), dev)
# -

# And this was all of the magic! The `EmbeddingKernel` class took care of providing us with a circuit that calculates the overlap. Before we can take a look at the kernel values we have to provide values for the variational parameters. We will initialize them such that the ansatz circuit has $28$ layers to be able to capture the full MNIST image.

init_params = random_params(N_WIRES, N_LAYERS)

# Now we can have a look at the kernel value between the first and the second datapoint:

print("The kernel value between the first and second datapoint is {:.3f}".format(k(train_X0[0], train_X1[0], init_params)))

# The mutual kernel values between all elements of the dataset from the _kernel matrix_. We can calculate it via the `square_kernel_matrix` method, which will be used in fit() later on:

K_init = k.square_kernel_matrix(X, init_params)

# ## Using the Quantum Embedding Kernel for predictions
#
# The quantum kernel alone can not be used to make predictions on a dataset, becaues it essentially just a tool to measure the similarity between two datapoints. To perform an actual prediction we will make use of scikit-learns support vector classifier (SVC). 

from sklearn.svm import SVC

# We will compute the kernel matrix needed for the `SVC` by hand, which is why we have to put `kernel="precomputed"` as the argument. Note that this does not train the parameters in our circuit but it trains the SVC on the kernel matrix with the given labels. 

svm = SVC(kernel="precomputed").fit(K_init, y)

# To see how well our classifier performs we will measure what percentage of the training set it classifies correctly.

print("The accuracy of a kernel with random parameters on the training set is {:.3f}".format(
    1 - np.count_nonzero(svm.predict(K_init) - y) / len(y))
)

# ## Evaluating performance on test data
#
# Now we will compare this to the performance on unseen data. To this end, we extract the next ten zeros and ones from the MNIST dataset:

# +
test_idx0 = np.argwhere(train_y == 0)[10:10+sample_size]
test_X0 = train_X[train_idx0].squeeze() * np.pi/255

test_idx1 = np.argwhere(train_y == 1)[10:10+sample_size]
test_X1 = train_X[train_idx1].squeeze() * np.pi/255

X_test = np.vstack([test_X0, test_X1])
y_test = np.hstack([[-1]*sample_size, [1]*sample_size])
# -

# To make a prediction, we have to compute the kernels between the test datapoints and the training datapoints. The `EmbeddingKernel` class offers a convenience method for this in the form of the `kernel_matrix` method.

K_pred = k.kernel_matrix(X_test, X, init_params)

# Now let's check how the kernel performs on the unseen data:

print("The test accuracy of a kernel with random parameters is {:.3f}".format(
    1 - np.count_nonzero(svm.predict(K_pred) - y_test) / len(y_test))
)
