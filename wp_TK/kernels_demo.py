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

# -

# # The data

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
# -

print("The training data is as follows:")
plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", label=1)
plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", label=-1)
plt.legend()

# Before diving into the code below, I'll plot the decision boundary to give an intuition as to what we are working towards. Hence; ignore the following code-block, as we'll get back to it

# +
n_qubits = len(X_train[0]) # -> equals number of features
dev_kernel = qml.device("default.qubit", wires=n_qubits)

projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """The quantum kernel."""
    AngleEmbedding(x1, wires=range(n_qubits))
    qml.inv(AngleEmbedding(x2, wires=range(n_qubits)))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

# function to compute the kernel using the quantum machine
# kernel computed by k(a,b) for all A and B
def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b) for b in B] for a in A])

# actually computer the kernel (on the quantum machine) using the scikit framework
svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)
# -

print("A plot of the decision boundary by inputting an input set that is uniformly spread")
print("Here we see the non-linear boundaries")
sweep = False
if sweep:

    precision = 10 # higher is preciser and more compute time
    # 10: about a minute
    # 100: about 100 minutes
    
    # create a dummy dataset that uniformly spans the input space
    X_dummy = []
    for i in range(-precision,precision):
        for j in range(-precision,precision):
            X_dummy.append([np.pi*i/precision,np.pi*j/precision])
    X_dummy = np.asarray(X_dummy)
    print(len(X_dummy))

    # predict (about a minute on my laptop)
    y_dummy = svm.predict(X_dummy)

    # plot in order to observe the decision boundary
    plt.scatter(X_dummy[np.where(y_dummy == 1)[0],0], X_dummy[np.where(y_dummy == 1)[0],1], color="b")
    plt.scatter(X_dummy[np.where(y_dummy == -1)[0],0], X_dummy[np.where(y_dummy == -1)[0],1], color="r")



# +
print("Now how does it create this decision boundary?")
print("It does so by the use of support vectors. For our dataset they are the black points in the following plot")

# some useful parameters:
sv_indices = svm.support_

plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", label="1")
plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", label="-1")
plt.scatter(X_train[sv_indices,0], X_train[sv_indices,1], color="k", label="support vectors")
plt.legend()


# +
print("In particular, it performs a weighted sum as follows: (see code)")
alpha = svm.dual_coef_[0] # the dual vectors
sv_indices = svm.support_ # the indices for the support vectors in the training data
b = svm.intercept_ # the bias
k = kernel # the kernel function
sample = 0 # which sample we want to classify for

result = np.sum([alpha[i] * k(X_train[sv_indices[i]], X_train[sample]) for i in range(len(alpha))]) +b
check = svm.decision_function([X_train[sample]])
print("Which results in", result[0], "which should be equal to", check[0])
print("The only thing left to is to sign it:", np.sign(result)[0], "to match the prediction:", y_train[sample])

# -

# However, here comes the whole trick (yes, it is literally called the kernel trick):
# Instead of calculating the kernel on the 2 datapoints, the data can be embedded and the norm taking (did I get this correct?)
# ![image.png](attachment:image.png)

# +
print("Now, how does one train such a model?")
print("In the case of the tutorial, one has to calculate the kernel between every element in X_train (see code comments)")

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

print("as far as I can see, no further training happend. Is this correct?")
print("y_train is purely fed to have it ready for pred(), but not used during training?")


# -

def validate(model, X, y_true):
    y_pred = model.predict(X)
    errors = np.sum(np.abs([(y_true[i] - y_pred[i])/2 for i in range(len(y_true))]))
    return (len(y_true)-errors)/len(y_true)
validate(svm, X_test, y_test)

# # Comparison to classical

# +
print("Now let's take a look at the classical alternatives. First the linear classifier:")


# Create a SVC classifier using a linear kernel
svm_lin = SVC(kernel='linear', C=1, random_state=0)
# Train the classifier
svm_lin.fit(X_train, y_train)

sweep = True
if sweep:

    precision = 100 # higher is preciser and more compute time
    
    # create a dummy dataset that uniformly spans the input space
    X_dummy = []
    for i in range(-precision,precision):
        for j in range(-precision,precision):
            X_dummy.append([np.pi*i/precision,np.pi*j/precision])
    X_dummy = np.asarray(X_dummy)
    print(len(X_dummy))

    # predict (about a minute on my laptop)
    y_dummy = svm_lin.predict(X_dummy)

    # plot in order to observe the decision boundary
    plt.scatter(X_dummy[np.where(y_dummy == 1)[0],0], X_dummy[np.where(y_dummy == 1)[0],1], color="b")
    plt.scatter(X_dummy[np.where(y_dummy == -1)[0],0], X_dummy[np.where(y_dummy == -1)[0],1], color="r")



# +
print("Now the rbf classifier:")


# Create a SVC classifier using a linear kernel
svm_rbf = SVC(kernel='rbf', random_state=0, gamma=1, C=1.5)
# Train the classifier
svm_rbf.fit(X_train, y_train)

sweep = True
if sweep:

    precision = 100 # higher is preciser and more compute time
    
    # create a dummy dataset that uniformly spans the input space
    X_dummy = []
    for i in range(-precision,precision):
        for j in range(-precision,precision):
            X_dummy.append([np.pi*i/precision,np.pi*j/precision])
    X_dummy = np.asarray(X_dummy)
    print(len(X_dummy))

    # predict (about a minute on my laptop)
    y_dummy = svm_rbf.predict(X_dummy)

    # plot in order to observe the decision boundary
    plt.scatter(X_dummy[np.where(y_dummy == 1)[0],0], X_dummy[np.where(y_dummy == 1)[0],1], color="b")
    plt.scatter(X_dummy[np.where(y_dummy == -1)[0],0], X_dummy[np.where(y_dummy == -1)[0],1], color="r")


# -

# So what's the deal?
# My guess is that the particular kernel in the article (or at least the one from the papers) is computationally more feasible on the quantum computer vs classically. But this does not mean it is always a better choise. As far as I remember, it is unknown when to used which kernel. I believe the current mantra is that the RBF has some very generic assumptions that are met by most datasets, and is therefor the default. So now the million dollar question: can we/they show this kernel is not only better to compute on a QPU, but also that this kernel has cases in which it is better than rbf?
