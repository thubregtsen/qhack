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
train_X1 = train_X[train_idx1].squeeze()

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

dev = qml.device("lightning.qubit", wires=7)
wires = list(range(7))
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x.flatten(), params, wires), dev)

# And this was all of the magic! The `EmbeddingKernel` class took care of providing us with a circuit that calculates the overlap. Before we can take a look at the kernel values we have to provide values for the variational parameters. We will initialize them such that the ansatz circuit has $6$ layers.

init_params = random_params(7, 4*28)

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


# We also want to see what kinds of decision boundaries the classifier realizes. To this end we will introduce a second helper method.

def plot_decision_boundaries(classifier, ax, N_gridpoints=14):
    _xx, _yy = np.meshgrid(np.linspace(-1, 1, N_gridpoints), np.linspace(-1, 1, N_gridpoints))

    _zz = np.zeros_like(_xx)
    for idx in np.ndindex(*_xx.shape):
        _zz[idx] = classifier.predict(np.array([_xx[idx], _yy[idx]])[np.newaxis,:])

    plot_data = {'_xx' : _xx, '_yy' : _yy, '_zz' : _zz}
    ax.contourf(_xx, _yy, _zz, cmap=mpl.colors.ListedColormap(['#FF0000', '#0000FF']), alpha=.2, levels=[-1, 0,  1])            
    dataset.plot(ax)
    
    return plot_data


# With that done, let's have a look at the decision boundaries for our initial classifier:

init_plot_data = plot_decision_boundaries(svm, plt.gca())

# We see that we can correctly classify the outer structure of the dataset, but our classifier still struggles with the inner points. But we have a circuit with many variational parameters, so it is reasonable to believe that we can improve the accuracy of our kernel based classification.

# ## Training the Quantum Embedding Kernel
#
# To be able to train the Quantum Embedding Kernel we need some measure of how well it fits the dataset in question. Re-training the SVM for every small change in the variational parameters and comparing the accuracy is no solution because it is very resource intensive and as the accuracy is a discrete quantity you would not be able to detect small improvements. 
#
# We can, however, resort to a more specialized measure, the _kernel-target alignment_ [1]. It is a measure that compares the similarity predicted by the quantum kernel to the actual labels of the training data. It is based on _kernel alignment_, a similiarity measure between two kernels with given kernel matrices $K_1$ and $K_2$:
#
# $$
# \operatorname{KA}(K_1, K_2) = \frac{\operatorname{Tr}(K_1 K_2)}{\sqrt{\operatorname{Tr}(K_1^2)\operatorname{Tr}(K_2^2)}}
# $$
#
# Seen from a more theoretical side, this is nothing else as the cosine of the angle between the kernel matrices $K_1$ and $K_2$ seen as vectors in the space of matrices with the Hilbert-Schmidt- (or Frobenius-) scalar product $\langle A, B \rangle = \operatorname{Tr}(A^T B)$.
#
# The training data enters picture by defining a kernel that expresses the labelling in the vector $\boldsymbol{y}$ by assigning the product of the respective labels as the kernel function
#
# $$
# k_{\boldsymbol{y}}(\boldsymbol{x}_i, \boldsymbol{x}_j) = y_i y_j
# $$
#
# The assigned kernel is thus $+1$ if both datapoints lie in the same class and $-1$ otherwise. The kernel matrix for this kernel is simply given by the outer product $\boldsymbol{y}\boldsymbol{y}^T$. The kernel-target alignment is then defined as the alignment of the kernel matrix generated by the quantum kernel and $\boldsymbol{y}\boldsymbol{y}^T$:
#
# $$
#     \operatorname{KTA}_{\boldsymbol{y}}(K) 
#     = \frac{\operatorname{Tr}(K \boldsymbol{y}\boldsymbol{y}^T)}{\sqrt{\operatorname{Tr}(K^2)\operatorname{Tr}((\boldsymbol{y}\boldsymbol{y}^T)^2)}} 
#     = \frac{\boldsymbol{y}^T K \boldsymbol{y}}{\sqrt{\operatorname{Tr}(K^2)} N}
# $$
#
# where $N$ is the number of elements in $\boldsymbol{y}$.
#
# In summary, the kernel-target alignment effectively captures how well the kernel you chose reproduces the actual similarities of the data. It is, however, only a necessary but not a sufficient criterion for a good performance of the kernel [1].
#
# Let's now come back to the actual implementation. PennyLane's `EmbeddingKernel` class allows you to easily evaluate the kernel target alignment:

print("The kernel-target-alignment for our dataset with random parameters is {:.3f}".format(
    k.target_alignment(dataset.X, dataset.Y, init_params))
)

# Now let's code up an optimization loop and improve this. To this end, we will make use of regular gradient descent optimization. To speed up the optimization we will sample smaller subsets of the data at each step, we choose $4$ datapoints at random. Remember that PennyLane's inbuilt optimizer works to _minimize_ the cost function that is given to it, which is why we have to multiply the kernel target alignment by $-1$ to actually _maximize_ it in the process. 

# +
params = init_params
opt = qml.GradientDescentOptimizer(2.5)

for i in range(500):
    subset = np.random.choice(list(range(len(dataset.X))), 4)
    params = opt.step(lambda _params: -k.target_alignment(dataset.X[subset], dataset.Y[subset], _params), params)
    
    if (i+1) % 50 == 0:
        print("Step {} - Alignment = {:.3f}".format(i+1, k.target_alignment(dataset.X, dataset.Y, params)))
# -

# Now let's build a second support vector classifier with the trained kernel:

svm_trained = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, params)).fit(dataset.X, dataset.Y)

# Now we expect to see that the accuracy has improved:

print("The accuracy of a kernel with trained parameters is {:.3f}".format(accuracy(svm_trained, dataset.X, dataset.Y)))

# Very well we now get perfect classification! We also expect that the decision boundaries of our classifier captures the nature of the dataset better, so let's check that:

trained_plot_data = plot_decision_boundaries(svm_trained, plt.gca())

# With this, we have seen that training our Quantum Embedding Kernel indeed yields not only improved accuracy but also more reasonable decision boundaries. In that sense, kernel training allows us to adjust the kernel to the dataset.

# ### References
#
# [1] Wang, Tinghua, Dongyan Zhao, and Shengfeng Tian. "An overview of kernel alignment and its applications." _Artificial Intelligence Review_ 43.2 (2015): 179-192.
