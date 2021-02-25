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

# # Quantum Embedding Kernels with PennyLane's kernels module
#
# _Authors: Peter-Jan Derks, Paul Fährmann, Elies Gil-Fuster, Tom Hubregtsen, Johannes Jakob Meyer and David Wierichs_
#
# Kernel methods are one of the cornerstones of classical machine learning. To understand what a kernel method does we first look at one of the possibly simplest methods to assign class labels to datapoints: linear classification.
#
# **TODO: Add intuitive explanation of kernel methods as in the paper**
#
# In this work, we will be concerned with _Quantum Embedding Kernels (QEKs)_, i.e. kernels that arise from embedding data into a quantum state. We formalize this by considering a quantum circuit $U(\boldsymbol{x})$ that embeds the datapoint $\boldsymbol{x}$ into the state
#
# $$
# |\psi(\boldsymbol{x})\rangle = U(\boldsymbol{x}) |0 \rangle.
# $$
#
# The kernel value is then given by the _overlap_ of the associated embedded quantum states
#
# $$
# k(\boldsymbol{x}, \boldsymbol{y}) = | \langle\psi(\boldsymbol{x})|\psi(\boldsymbol{y})\rangle|^2.
# $$

# ## A toy problem
#
# In this demonstration, we will treat a toy problem that showcases the inner workings of our approach. We will create the `DoubleCake` dataset. To do so, we first have to do some imports:

import pennylane as qml
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class DoubleCake:
    def _make_circular_data(self): 
        """Generate datapoints arranged in an even circle."""
        center_indices = np.array(range(0, self.num_sectors))
        sector_angle = 2*np.pi / self.num_sectors
        angles = (center_indices + 0.5) * sector_angle
        x = 0.7 * np.cos(angles)
        y = 0.7 * np.sin(angles)
        labels = 2 * np.remainder(np.floor_divide(angles, sector_angle), 2)- 1 
        
        return x, y, labels

    def __init__(self, num_sectors):
        self.num_sectors = num_sectors
        
        x1, y1, labels1 = self._make_circular_data()
        x2, y2, labels2 = self._make_circular_data()

        # x and y coordinates of the datapoints
        self.x = np.hstack([x1, .5 * x2])
        self.y = np.hstack([y1, .5 * y2])
        
        # Canonical form of dataset
        self.X = np.vstack([self.x, self.y]).T
        
        self.labels = np.hstack([labels1, -1 * labels2])
        
        # Canonical form of labels
        self.Y = self.labels.astype(int)

    def plot(self, ax, show_sectors=False):
        ax.scatter(self.x, self.y, c=self.labels, cmap=mpl.colors.ListedColormap(['#FF0000', '#0000FF']), s=10)
        sector_angle = 360/self.num_sectors
        
        if show_sectors:
            for i in range(self.num_sectors):
                color = ['#FF0000', '#0000FF'][(i % 2)]
                other_color = ['#FF0000', '#0000FF'][((i + 1) % 2)]
                ax.add_artist(mpl.patches.Wedge((0, 0), 1, i * sector_angle, (i+1)*sector_angle, lw=0, color=color, alpha=0.1, width=.5))
                ax.add_artist(mpl.patches.Wedge((0, 0), .5, i * sector_angle, (i+1)*sector_angle, lw=0, color=other_color, alpha=0.1))
                ax.set_xlim(-1, 1)

        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.axis("off")


# Let's now have a look at our dataset. In our example, we will work with 6 sectors:

# +
dataset = DoubleCake(6)

dataset.plot(plt.gca(), show_sectors=True)


# -

# ## Defining a Quantum Embedding Kernel
#
# PennyLane's `kernels` module allows for a particularly simple implementation of Quantum Embedding Kernels. The first ingredient we need for this is an _ansatz_ that represents the unitary $U(\boldsymbol{x})$ we use for embedding the data into a quantum state. 

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

# We are now in a place where we can create the embedding. Together with the ansatz we only need a device to run the quantum circuit on. For the purposes of this tutorial we will use PennyLane's `lightning.qubit` device with 5 wires.
#
# To add another interesting twist, we will not repeatedly input the different datapoints but extract random linear combinations. This is realized by choosing a matrix $W$ whose entries are randomly sampled from the normal distribution. We have $2$ data dimensions but want to expand them to $30$ different embedding features. We therefore construct a matrix with shape $(2, 30)$ so that the matrix-vector product $\boldsymbol{x}W$ is a vector with $30$ entries.

dev = qml.device("lightning.qubit", wires=5)
wires = list(range(5))
W = np.random.normal(0, .7, (2, 30))
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x @ W, params, wires), dev)

# And this was all of the magic! The `EmbeddingKernel` class took care of providing us with a circuit that calculates the overlap. Before we can take a look at the kernel values we have to provide values for the variational parameters.

init_params = random_params(5, 6)

# Now we can have a look at the kernel value between the first and the second datapoint:

k(dataset.X[0], dataset.X[1], init_params)

# The mutual kernel values between all elements of the dataset form the _kernel matrix_. We can inspect it via the `square_kernel_matrix` method:

# +
K_init = k.square_kernel_matrix(dataset.X, init_params)

with np.printoptions(precision=3, suppress=True):
    print(K_init)
# -

# ## Using the Quantum Embedding Kernel for predictions
#
# The quantum kernel alone can not be used to make predictions on a dataset, becaues it essentially just a tool to measure the similarity between two datapoints. To perform an actual prediction we will make use of scikit-learns support vector classifier (SVC). 

from sklearn.svm import SVC

# The `SVC` class expects a function that maps two sets of datapoints to the corresponding kernel matrix. This is provided by the `kernel_matrix` property of the `EmbeddingKernel` class, we only have to use a lambda construction to include our parameters.

svm = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, init_params)).fit(dataset.X, dataset.Y)


# To see how well our classifier performs we will measure what percentage it classifies correctly.

def accuracy(classifier, X, Y_target):
    return np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)


print("The accuracy of a kernel with random parameters is {:.3f}".format(accuracy(svm, dataset.X, dataset.Y)))

# We also want to see what kinds of decision boundaries the classifier realizes. To this end we will introduce a second helper method.


