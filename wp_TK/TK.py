# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

import matplotlib.pyplot as plt

import jupytext

np.random.seed(42)

# +
# load the data
X, y = load_iris(return_X_y=True) 

print("The dataset contains X and y, each of length", len(X))
print("X contains", len(X[0]), "features")
print("y contains the following classes", np.unique(y))

# pick inputs and labels from the first two classes only,
# corresponding to the first 100 samples
# -> meanig y now consists of 2 classes: 0, 1; still stored in order, balanced 50:50
X = X[:100,0:2]
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

# +
n_qubits = len(X_train[0]) # -> equals number of features
dev_kernel = qml.device("default.qubit", wires=n_qubits)

projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

# The actual kernel
@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """The quantum kernel."""
    AngleEmbedding(x1, wires=range(n_qubits))
    qml.inv(AngleEmbedding(x2, wires=range(n_qubits)))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

# a helper method that calculates the k(a,b) for all in A and B
def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b) for b in B] for a in A])

# the actual call that feeds X_train into kernel_matrix(A,B) and calculated the distances between all points
svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)
# -

#dim = 3
dim = len(X_train)
P = np.asarray([y_train[i]*y_train[j]*kernel(X_train[i], X_train[j]) for i in range(dim) for j in range(dim)]).reshape((dim, dim))

P.shape


