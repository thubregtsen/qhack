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

# ### Initialization and circuit definitions

# +
import pennylane as qml
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets

np.random.seed(21) # sorry, 42 did not build a nice dataset


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

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)
# +
features = 2
width = 5
depth = 16

# init device
dev = qml.device("default.qubit", wires=width)
wires = list(range(width))

# init the embedding kernel
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x, params, wires), dev)


# -


# ### Alternative data set
#
# Two pairs of concenctric circles centered at +/- 1 and small radius .4, big radius .8.
# Classes are inverted right and left

def datagen (n_train, n_test):
    # generate data in two circles
    n_part = int(n_train/2)
    n_test = int(n_test/2)
    i = 0
    X = []
    X_ = []
    y = []
    y_ = []
    while (i<n_part):
        x1 = np.random.uniform(-.8,.8)
        x2 = np.random.uniform(-.8,.8)
        if((x1)*(x1) + x2*x2 < .64):
            i+=1
            X.append([1+x1,x2])
            if(x1*x1 + x2*x2 < .16):
                y.append(1)
            else:
                y.append(-1)
    
    i=0
    while(i<n_part):
        x1 = np.random.uniform(-.8,.8)
        x2 = np.random.uniform(-.8,.8)
        if(x1*x1 + x2*x2 <.64):
            i+=1
            X.append([x1-1,x2])
            if(x1*x1 + x2*x2 < .16):
                y.append(-1)
            else:
                y.append(1)
    
    i = 0
    while (i<n_test):
        x1 = np.random.uniform(-.8,.8)
        x2 = np.random.uniform(-.8,.8)
        if(x1*x1 + x2*x2 < .64):
            i+=1
            X_.append([1+x1,x2])
            if(x1*x1 + x2*x2 < .16):
                y_.append(1)
            else:
                y_.append(-1)
    
    i=0
    while(i<n_test):
        x1 = np.random.uniform(-.8,.8)
        x2 = np.random.uniform(-.8,.8)
        if(x1*x1 + x2*x2 <.64):
            i+=1
            X_.append([x1-1,x2])
            if(x1*x1 + x2*x2 < .16):
                y_.append(-1)
            else:
                y_.append(1)
            
    return X,y, X_,y_


X_train ,y_train, X_test, y_test = datagen(20,160)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

X_train.shape, X_test.shape

# plot for visual inspection
print("The training data is as follows:")
plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker=".", label="train, 1")
plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker=".", label="train, -1")
#print("The test data is as follows:")
plt.scatter(X_test[np.where(y_test == 1)[0],0], X_test[np.where(y_test == 1)[0],1], color="b", marker="x", label="test, 1")
plt.scatter(X_test[np.where(y_test == -1)[0],0], X_test[np.where(y_test == -1)[0],1], color="r", marker="x", label="test, -1")
plt.ylim([-1, 1])
plt.xlim([-2, 2])
plt.legend()
plt.show()

# ### Assessment of random initialization

# make various runs with random parameters to see what range the results can be in
acc_log = []
params_log = []
for i in range(5):
    ## choose random params for the kernel
    params = random_params(width, depth)
    #print(params)
    ## fit the SVM on the training data
    svm_untrained_kernel = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, params)).fit(X_train, y_train)
    ## evaluate on the test set
    untrained_accuracy = accuracy(svm_untrained_kernel, X_test, y_test)
    print("without kernel training accuracy", untrained_accuracy)
    acc_log.append(untrained_accuracy)
    params_log.append(params)
print("going with", acc_log[np.argmin(np.asarray(acc_log))])
params = params_log[np.argmin(np.asarray(acc_log))]

print("Untrained accuracies:", acc_log)

plotting =  True

# ### Gradient Ascent Alignment

# evaluate the performance with trained parameters for the kernel
## train the kernel
opt = qml.GradientDescentOptimizer(.5)
for i in range(1000):
    subset = np.random.choice(list(range(len(X_train))), 4)
    params = opt.step(lambda _params: -k.target_alignment(X_train[subset], y_train[subset], _params), params)
    
    if (i+1) % 50 == 0:
        print("Step {} - Alignment on train = {:.3f}".format(i+1, k.target_alignment(X_train, y_train, params)))
# ### Train SVM

# +
## fit the SVM on the train set
svm_trained_kernel = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, params)).fit(X_train, y_train)
p_train = svm_trained_kernel.predict(X_train)
p_test = svm_trained_kernel.predict(X_test)

## evaluate the accuracy on the test set
trained_accuracy = accuracy(svm_trained_kernel, X_train, y_train)
test_accuracy = accuracy(svm_trained_kernel, X_test, y_test)
print("with kernel training accuracy on train", trained_accuracy)
print("with kernel training accuracy on test", test_accuracy)
# -
# ### Plot results

# +
start = -2
stop = 2
num = 20

X_dummy = []

inc = (stop-start)/num

for i in range(num):
    for j in range(num):
        X_dummy.append([start + i*inc,start + j*inc])
# -

y_dummy = svm_trained_kernel.predict(X_dummy)

X_dummy = np.asarray(X_dummy)
y_dummy = np.asarray(y_dummy)

plt.scatter(X_dummy[np.where(y_dummy == 1)[0],0], X_dummy[np.where(y_dummy == 1)[0],1], color="b", marker=".",label="dummy, 1")
plt.scatter(X_dummy[np.where(y_dummy == -1)[0],0], X_dummy[np.where(y_dummy == -1)[0],1], color="r", marker=".",label="dummy, -1")
plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker="+", label="train, 1")
plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker="+", label="train, -1")
plt.scatter(X_test[np.where(y_train == 1)[0],0], X_test[np.where(y_train == 1)[0],1], color="b", marker="x", label="test, 1")
plt.scatter(X_test[np.where(y_train == -1)[0],0], X_test[np.where(y_train == -1)[0],1], color="r", marker="x", label="test, -1")
plt.ylim([-1, 1])
plt.xlim([-2, 2])
plt.legend()

