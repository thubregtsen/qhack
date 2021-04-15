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

np.random.seed(42)


# +
# the Ansatz
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
# +
# define accuracy
def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)

def accuracy_predict(Y_predict, Y_target):
    return 1 - np.count_nonzero(Y_predict - Y_target) / len(Y_target)
# +
# circuit and device parameters
features = 2
width = 3
depth = 3

# init device
dev = qml.device("default.qubit", wires=width)
wires = list(range(width))

# init the embedding kernel
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x, params, wires), dev)


# -


# ### Symmetric donuts data set
#
# Two pairs of concenctric circles centered at +/- 1 and small radius $1/2$, big radius $1/\sqrt{2}$.
# Classes are inverted right and left

# generate data
def datagen (n_train, n_test):
    # generate data in two circles
    # the radii are chosen so that data is balanced
    n_part = int(n_train/2)
    n_test = int(n_test/2)
    i = 0
    X = []
    X_ = []
    y = []
    y_ = []

    # training set, right circle
    while (i<n_part):
        # sample uniformly from a square
        x1 = np.random.uniform(-.707,.707) # 0.707... = 0.5*\sqrt(2)
        x2 = np.random.uniform(-.707,.707)
        if((x1)*(x1) + x2*x2 < .5): # discard points outside of big radius
            i+=1
            X.append([1+x1,x2]) # center at (1,0)
            if(x1*x1 + x2*x2 < .25): # inner circle class 1
                y.append(1)
            else:
                y.append(-1) # outer circle class -1
    
    # training set, left circle
    i=0
    while(i<n_part):
        # sample uniformly from a square
        x1 = np.random.uniform(-.707,.707)
        x2 = np.random.uniform(-.707,.707)
        if(x1*x1 + x2*x2 <.5): # discard points outside of big radius
            i+=1
            # opposite than first iteration
            X.append([x1-1,x2]) # center at (-1,0)
            if(x1*x1 + x2*x2 < .25): # inner circle class -1
                y.append(-1)
            else:
                y.append(1) # outer circle class 1
    
    # test set, right circle
    i = 0
    while (i<n_test):
        # sample uniformly from a square
        x1 = np.random.uniform(-.707,.707) # 0.707... = 0.5*\sqrt(2)
        x2 = np.random.uniform(-.707,.707)
        if((x1)*(x1) + x2*x2 < .5): # discard points outside of big radius
            i+=1
            X.append([1+x1,x2]) # center at (1,0)
            if(x1*x1 + x2*x2 < .25): # inner circle class 1
                y.append(1)
            else:
                y.append(-1) # outer circle class -1
    
    # test set, left circle
    i=0
    while(i<n_test):
        # sample uniformly from a square
        x1 = np.random.uniform(-.707,.707)
        x2 = np.random.uniform(-.707,.707)
        if(x1*x1 + x2*x2 <.5): # discard points outside of big radius
            i+=1
            # opposite than first iteration
            X.append([x1-1,x2]) # center at (-1,0)
            if(x1*x1 + x2*x2 < .25): # inner circle class -1
                y.append(-1)
            else:
                y.append(1) # outer circle class 1
            
    return X,y, X_,y_


X_train, y_train, X_test, y_test = datagen(60,60)

#justnumpythings
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# sanity check
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
test_predict_log = []
train_predict_log = []
for i in range(5):
    ## choose random params for the kernel
    params = random_params(width, depth)
    ## fit the SVM on the training data
    svm_untrained_kernel = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, params)).fit(X_train, y_train)
    ## evaluate on the test set
    test_predict_log.append(svm_untrained_kernel.predict(X_test))
    train_predict_log.append(svm_untrained_kernel.predict(X_train))
    untrained_accuracy_test = accuracy_predict(test_predict_log[-1], y_test)
    untrained_accuracy_train = accuracy_predict(train_predict_log[-1], y_train)
    print("without kernel training accuracy on test", untrained_accuracy_test)
    print("without kernel training accuracy on train", untrained_accuracy_train)
    acc_log.append(untrained_accuracy_test)
    params_log.append(params)
print("going with", acc_log[np.argmin(np.asarray(acc_log))])

# select untrained kernel with lowest accuracy on test
params_random = params_log[np.argmin(np.asarray(acc_log))]

print("Untrained accuracies:", acc_log)

# plotting takes place at the end, but still
plotting =  True

# ### Gradient Ascent Alignment

# evaluate the performance with trained parameters for the kernel
## train the kernel
alignment = k.target_alignment(X_train, y_train, params)
alignment_log = [alignment]
print("Step 0 - Alignment on train = {:.3f}".format(alignment))
opt = qml.GradientDescentOptimizer(1)
for i in range(1000): # maxiter = 1000
    subset = np.random.choice(list(range(len(X_train))), 4)
    params = opt.step(lambda _params: -k.target_alignment(X_train[subset], y_train[subset], _params), params)
    
    if (i+1) % 50 == 0:
        alignment = k.target_alignment(X_train, y_train, params)
        alignment_log.append(alignment)
        print("Step {} - Alignment on train = {:.3f}".format(i+1, alignment))
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
# create grid for decision boundary and the contour plot
startx = -2
stopx = 2
starty = -1
stopy = 1
num = 20 #precision

X_dummy = []

incx = (stopx-startx)/num
incy = (stopy-starty)/num

for i in range(num):
    for j in range(num):
        X_dummy.append([startx + i*incx,starty + j*incy])
# -

# decision function is the actual value without the sign function
y_dummy = svm_trained_kernel.decision_function(X_dummy)

#justnumpyythings
X_dummy = np.asarray(X_dummy)
y_dummy = np.asarray(y_dummy)

# +
# prediction = sign(decision_function)
y_dummy_label = np.sign(y_dummy)

# plot for the decision boundary
plt.scatter(X_dummy[np.where(y_dummy_label == 1)[0],0], X_dummy[np.where(y_dummy_label == 1)[0],1], color="b", marker=".",label="dummy, 1")
plt.scatter(X_dummy[np.where(y_dummy_label == -1)[0],0], X_dummy[np.where(y_dummy_label == -1)[0],1], color="r", marker=".",label="dummy, -1")
plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker="+", label="train, 1")
plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker="+", label="train, -1")
plt.scatter(X_test[np.where(y_test == 1)[0],0], X_test[np.where(y_test == 1)[0],1], color="b", marker="x", label="test, 1")
plt.scatter(X_test[np.where(y_test == -1)[0],0], X_test[np.where(y_test == -1)[0],1], color="r", marker="x", label="test, -1")
plt.ylim([-1, 1])
plt.xlim([-2, 2])
plt.legend()
# -
# contour plot
plt.scatter(X_dummy[:,0], X_dummy[:,1],marker='s', s=140, c= y_dummy, alpha=0.9)
plt.colorbar()

# now for the randomly initialized kernel
y_dummy_random = svm_untrained_kernel.decision_function(X_dummy)
y_dummy_random_label = np.sign(y_dummy_random)

# decision boundary plot for untrained kernel
plt.scatter(X_dummy[np.where(y_dummy_random_label == 1)[0],0], X_dummy[np.where(y_dummy_random_label == 1)[0],1], color="b", marker=".",label="dummy, 1")
plt.scatter(X_dummy[np.where(y_dummy_random_label == -1)[0],0], X_dummy[np.where(y_dummy_random_label == -1)[0],1], color="r", marker=".",label="dummy, -1")
plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker="+", label="train, 1")
plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker="+", label="train, -1")
plt.scatter(X_test[np.where(y_test == 1)[0],0], X_test[np.where(y_test == 1)[0],1], color="b", marker="x", label="test, 1")
plt.scatter(X_test[np.where(y_test == -1)[0],0], X_test[np.where(y_test == -1)[0],1], color="r", marker="x", label="test, -1")
plt.ylim([-1, 1])
plt.xlim([-2, 2])
plt.legend()

# contour plot for untrained kernel
plt.scatter(X_dummy[:,0], X_dummy[:,1],marker='s', s=140, c= y_dummy_random, alpha=0.9)
plt.colorbar()

# ### Helper functions for data saving and loading
#
# They're markdown cells now so no files are overwritten by accident, just turn them into code and hack away!

# filename = "dataset_symmetricdonuts.npy"
# with open(filename, 'wb') as f:
#     np.save(f, X_dummy)
#     np.save(f, y_dummy_label)
#     np.save(f, y_dummy)
#     np.save(f, y_dummy_random_label)
#     np.save(f, y_dummy_random)
#     np.save(f, X_train)
#     np.save(f, y_train)
#     np.save(f, X_test)
#     np.save(f, y_test)

# with open(filename, 'rb') as f:
#     X_dummy_c = np.load(f)
#     y_dummy_label_c = np.load(f)
#     y_dummy_c = np.load(f)
#     y_dummy_random_label_c = np.load(f)
#     y_dummy_random_c = np.load(f)
#     X_train_c = np.load(f)
#     y_train_c = np.load(f)
#     X_test_c = np.load(f)
#     y_test_c = np.load(f)

# filename = "parameters_symmetricdonuts.npy"
# with open(filename, 'wb') as f:
#     np.save(f, params)
#     np.save(f, params_random)

# with open(filename, 'rb') as f:
#     params_c = np.load(f)
#     params_random_c = np.load(f)

# filename = "alignment_symmetricdonuts.npy"
# with open(filename, 'wb') as f:
#     np.save(f, alignment_log)


