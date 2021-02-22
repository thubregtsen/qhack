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
dev_kernel = qml.device("qulacs.simulator", wires=n_qubits)

projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1


# -

def polarization(kernel, X_train, Y_train, kernel_args=(), samples=None, seed=None):
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
    if seed is None:
        seed = np.random.randint(0, 1000000)
    np.random.seed(seed)
    if samples is None:
        x = X_train
        y = Y_train
        samples = len(y)
    else:
        sampled = np.random.choice(list(range(len(Y_train))), samples)
        x = X_train[sampled]
        y = Y_train[sampled]
    
    P = 0
    # Only need to compute the upper right triangle of the kernel matrix and y_correl_matrix (they are symmetric)
    # Actually, the diagonal is usually going to be 1 (for y_correl it is for labels +-1), but we can see that later
    for i, (x1, y1) in enumerate(zip(x, y)):
        P += y1*y1 * kernel(x1, x1) # Usually will be 1
        for x2, y2 in zip(x[i+1:], y[i+1:]):
            P += 2 * y1 * y2 * kernel(x1, x2)
            
    return P


# +
# The actual kernel
@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """The quantum kernel."""
    global param # as opposed to changing sklearn.SVC
    AngleEmbedding(x1, wires=range(n_qubits))
    qml.CRX(param[0], wires=[0, 1])
    qml.inv(AngleEmbedding(x2, wires=range(n_qubits)))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

#def polarization_cost(param, X_train, y_train, kernel, samples=None, seed=None):
#    return polarization(kernel, X_train, y_train, kernel_args=[param], samples=samples, seed=seed)


# -


param = np.random.random(1)*2*np.pi
#polarization_cost(param, X_train, y_train, kernel)
polarization(kernel, X_train, y_train, kernel_args=[param])




# +

# a helper method that calculates the k(a,b) for all in A and B
def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b) for b in B] for a in A])

# the actual call that feeds X_train into kernel_matrix(A,B) and calculated the distances between all points
def validate(model, X, y_true):
    y_pred = model.predict(X)
    errors = np.sum(np.abs([(y_train[i] - y_pred[i])/2 for i in range(len(y_train))]))
    return (len(y_true)-errors)/len(y_true)


# +
svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)
param = np.random.random(1)*2*np.pi
print("random p; p untrained; kernel untrained", validate(svm, X_train, y_train))

svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)
print("random p; p untrained; kernel trained", validate(svm, X_train, y_train))

param = [0 for x in param]
print("p at zero; kernel untrained", validate(svm, X_train, y_train))

svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)
print("p at zero; kernel trained", validate(svm, X_train, y_train))


# -























