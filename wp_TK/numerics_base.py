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

# +
import pennylane as qml
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC

np.random.seed(42+1) # sorry, 42 did not build a nice dataset


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



def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)


# +
## create dataset
#samples = 20
#features = 2
### choose random input
#X = np.random.random((2*samples,features))
### choose random params for the kernel, which will be our ideal parameters
#ideal_params = random_params(5, 6)
## init the embedding kernel
#k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x, params, wires), dev)

##generate train data
#X_train = X[:samples]
### choose random y targets in a balanced way
#indices = np.arange(samples)
#np.random.shuffle(indices)
#indices = indices[:int(samples/2)] # uneven will get rounded
#y_init = np.ones((samples))
#y_init[indices] = -1
### fit the SVM
#svm_init_train = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, ideal_params)).fit(X_train, y_init)
### and determine what the SVM can actually predict with our ideal parameters
#y_train = svm_init_train.predict(X_train)
## do it again
#svm_init_train = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, ideal_params)).fit(X_train, y_train)
### and determine what the SVM can actually predict with our ideal parameters
#y_train = svm_init_train.predict(X_train)

##generate test data
#X_test = X[samples:]
### choose random y targets in a balanced way
#indices = np.arange(samples)
#np.random.shuffle(indices)
#indices = indices[:int(samples/2)] # uneven will get rounded
#y_init = np.ones((samples))
#y_init[indices] = -1
### fit the SVM
#svm_init_test = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, ideal_params)).fit(X_test, y_init)
### and determine what the SVM can actually predict with our ideal parameters
#y_test = svm_init_test.predict(X_test)
## do it again
#svm_init_test = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, ideal_params)).fit(X_test, y_test)
### and determine what the SVM can actually predict with our ideal parameters
#y_test = svm_init_test.predict(X_test)

#print("X_train:", X_train)
#print("y_train:", y_train)
#print("X_test:", X_test)
#print("y_test:", y_test)

## sanity check
#ideal_accuracy_train = accuracy(svm_init_train, X_train, y_train)
#ideal_accuracy_test = accuracy(svm_init_test, X_test, y_test)
#print("these 2 values should be high", ideal_accuracy_train, ideal_accuracy_test)

# -
def make_dataset(kernel, data_shape, data_domain=(0,1), N1=10, N2=10, lower_interval=(0.0, 0.3), upper_interval=(.7, .9), seed=None):
    retry_limit = 100000
    if seed is None:
        seed = np.random.uniform(*data_domain, data_shape)
    
    friends = [seed] # add a seed point
    enemies = []
    it = 0
    while (len(friends) < N1 or len(enemies) < N2) and it < retry_limit:
        datapoint = np.random.uniform(*data_domain, data_shape)
        
        val = kernel(friends[0], datapoint)
        
        if val >= upper_interval[0] and val <= upper_interval[1] and len(friends) < N1:
            print("Friend!", end=" ")
            friends.append(datapoint)
        elif val >= lower_interval[0] and val <= lower_interval[1] and len(enemies) < N2:
            print("Foe!", end=" ")
            enemies.append(datapoint)
            
        it += 1
            
    if it == retry_limit:
        print("DIDN'T SUCCEED TO BUILD A DATASET IN", retry_limit, "ITERATIONS.")
    
    print("\nTook ", it, " iterations.")
        
    # returns X and y
    return np.vstack([friends, enemies]), np.hstack([[1]*len(friends), [-1]*len(enemies)])            


# +
samples = 10
features = 10
width = 5
depth = 20


dev = qml.device("default.qubit", wires=width)
wires = list(range(width))

# init the embedding kernel
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x, params, wires), dev)
## choose random params for the kernel, which will be our ideal parameters
ideal_params = random_params(width, depth) 

# -

X, y = make_dataset(lambda x,y: k(x,y,ideal_params), (features,), N1 = 2*samples, N2=2*samples)
#X_test, y_test = make_dataset(lambda x,y: k(x,y,ideal_params), (samples,), seed=X_train[0], N1 = samples, N2 = samples)



# +
#K = k.square_kernel_matrix(X_train, ideal_params)

# +
#K[:,0]
#half_samples = int(samples/2)
# -







X_train = np.vstack([X[:samples], X[2*samples:3*samples]])
y_train = np.hstack([y[:samples], y[2*samples:3*samples]])
X_test = np.vstack([X[samples:2*samples], X[3*samples:]])
y_test = np.hstack([y[samples:2*samples], y[3*samples:]])
#print(X_train)
#print(y_train)
#print(X_test)#
#print(y_test)


# evaluate the performance with random parameters for the kernel
## choose random params for the kernel
params = random_params(width, depth)
## fit the SVM on the training data
svm_untrained_kernel = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, params)).fit(X_train, y_train)
## evaluate on the test set
untrained_accuracy = accuracy(svm_untrained_kernel, X_test, y_test)
print("without kernel training accuracy", untrained_accuracy)

# evaluate the performance with trained parameters for the kernel
## train the kernel
opt = qml.GradientDescentOptimizer(0.5)
for i in range(500):
    subset = np.random.choice(list(range(len(X_train))), 4)
    params = opt.step(lambda _params: -k.target_alignment(X_train[subset], y_train[subset], _params), params)
    
    if (i+1) % 50 == 0:
        print("Step {} - Alignment on train = {:.3f}".format(i+1, k.target_alignment(X_train, y_train, params)))

## fit the SVM on the train set
svm_trained_kernel = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, params)).fit(X_train, y_train)
## evaluate the accuracy on the test set
trained_accuracy = accuracy(svm_untrained_kernel, X_test, y_test)
print("with kernel training accuracy on test", trained_accuracy)

## sanity check, fit on train with IDEAL parameters
svm_ideal_kernel = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, ideal_params)).fit(X_train, y_train)
## evaluate on train
ideal_accuracy = accuracy(svm_ideal_kernel, X_train, y_train)
## should be high
print("ideal accuracy", ideal_accuracy)






















































































