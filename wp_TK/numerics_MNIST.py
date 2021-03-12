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
from sklearn import datasets
import matplotlib as mpl
from keras.datasets import mnist

np.random.seed(42)


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
features = 2
width = 10
depth = 320 


dev = qml.device("default.qubit", wires=width)
wires = list(range(width))

# init the embedding kernel
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x, params, wires), dev)
# +
(train_X, train_y), (test_X, test_y) = mnist.load_data()

sample_size = 1000 # will be x+x

train_idx0 = np.argwhere(train_y == 0)[:4*sample_size]
train_X0 = train_X[train_idx0].squeeze() * np.pi/255

train_idx1 = np.argwhere(train_y == 1)[:4*sample_size]
train_X1 = train_X[train_idx1].squeeze() * np.pi/255

X = np.vstack([train_X0, train_X1])
y = np.hstack([[-1]*sample_size, [1]*sample_size])

X_train = np.vstack([train_X0[:sample_size], train_X1[:sample_size]])
X_test = np.vstack([train_X0[2*sample_size:3*sample_size], train_X1[2*sample_size:3*sample_size]])
y_train = np.hstack([[-1]*sample_size, [1]*sample_size])
y_test = np.hstack([[-1]*sample_size, [1]*sample_size])

# +
gs = mpl.gridspec.GridSpec(2, sample_size) 
fig = plt.figure(figsize=(8,2))

print(y_train)
for j in range(sample_size):
    ax=plt.subplot(gs[0, j])
    plt.imshow(X_train[j], cmap=plt.get_cmap('gray'))
    ax.axis("off")
    
    ax=plt.subplot(gs[1, j])
    plt.imshow(X_train[sample_size+j], cmap=plt.get_cmap('gray'))
    ax.axis("off")    
plt.show()

print(y_test)
gs = mpl.gridspec.GridSpec(2, sample_size) 
fig = plt.figure(figsize=(8,2))
for j in range(sample_size):
    ax=plt.subplot(gs[0, j])
    plt.imshow(X_test[j], cmap=plt.get_cmap('gray'))
    ax.axis("off")
    
    ax=plt.subplot(gs[1, j])
    plt.imshow(X_test[sample_size+j], cmap=plt.get_cmap('gray'))
    ax.axis("off")  
plt.show()
# -


np.asarray([x.flatten() for x in X_train]).shape

X_train[sample_size+2][5]

np.where(X_train[sample_size+2]!=0)

# +
x_0, x_1 = np.where(train_X0[0]>=0.95)
sample_indices = np.random.randint(low=0, high=len(x_0), size=3)
zero_samples = [x_0[sample_indices], x_1[sample_indices]]

for i in range(1,1000):
    x_0, x_1 = np.where(train_X0[i]>=0.95)
    sample_indices = np.random.randint(low=0, high=len(x_0), size=3)
    zero_samples[0] = np.hstack([zero_samples[0], x_0[sample_indices]])
    zero_samples[1] = np.hstack([zero_samples[1], x_1[sample_indices]])

plt.scatter(zero_samples[0], zero_samples[1])
plt.show()

#train_X0
x_0, x_1 = np.where(train_X1[0]>=0.95)
sample_indices = np.random.randint(low=0, high=len(x_0), size=3)
one_samples = [x_0[sample_indices], x_1[sample_indices]]

for i in range(2,1000):
    x_0, x_1 = np.where(train_X1[i]>=0.95)
    sample_indices = np.random.randint(low=0, high=len(x_0), size=3)
    one_samples[0] = np.hstack([one_samples[0], x_0[sample_indices]])
    one_samples[1] = np.hstack([one_samples[1], x_1[sample_indices]])

plt.scatter(one_samples[0], one_samples[1])
plt.show()

#plt.scatter(np.unique(zero_samples[0], one_samples[0]), np.unique(zero_samples[1], one_samples[1]))
#plt.show()



# +
zeros = np.unique(np.asarray(zero_samples)[:].T, axis=0)
ones = np.unique(np.asarray(one_samples)[:].T, axis=0)
distinct_zeros = []
distinct_ones = []
duplicates = []
# find unique zeros and duplicates
for sample in zeros:
    first_index = np.where(ones[:,0] == sample[0])
    #print(first_index)
    if(len(first_index)>0):
        second_index = np.where(ones[first_index][:,1] == sample[1])
        if(len(second_index[0])>0):
            #print("Present")
            duplicates.append(sample)
        else:
            #print("Not present")
            distinct_zeros.append(sample)
    else:
        #print("not present")
        distinct_zeros.append(sample)

# find unique ones: 
for sample in ones:
    first_index = np.where(zeros[:,0] == sample[0])
    #print(first_index)
    if(len(first_index)>0):
        second_index = np.where(zeros[first_index][:,1] == sample[1])
        if(len(second_index[0])>0):
            continue
            #print("Present")
            #duplicates.append(sample)
        else:
            #print("Not present")
            distinct_ones.append(sample)
    else:
        #print("not present")
        distinct_ones.append(sample)
        
        
distinct_zeros = np.asarray(distinct_zeros).T/28
duplicates = np.asarray(duplicates).T/28
distinct_ones = np.asarray(distinct_ones).T/28

plt.scatter(distinct_zeros[0], distinct_zeros[1])
plt.title("Distinct zeros")
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
plt.scatter(duplicates[0], duplicates[1])
plt.title("Duplicates")
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
plt.scatter(distinct_ones[0], distinct_ones[1])
plt.title("Distinct ones")
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
#print(distinct_zeros)

not_zeros = np.hstack([distinct_ones, duplicates])
plt.title("Not zeros")
plt.scatter(not_zeros[0], not_zeros[1])
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

not_ones = np.hstack([distinct_zeros, duplicates])
plt.title("Not ones")
plt.scatter(not_ones[0], not_ones[1])
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
# -





# # Is it 0?

sample_size = 10
zero_indices = np.random.randint(low=0, high=len(distinct_zeros[0]), size=sample_size)
not_zero_indices = np.random.randint(low=0, high=len(not_zeros[0]), size=sample_size)
X_train = np.vstack([distinct_zeros.T[zero_indices], not_zeros.T[not_zero_indices]])
y_train = np.hstack([np.ones((sample_size)), -np.ones((sample_size))])
plt.scatter(X_train[np.where(y_train==1)][:,0], X_train[np.where(y_train==1)][:,1], label="zero")
plt.scatter(X_train[np.where(y_train==-1)][:,0], X_train[np.where(y_train==-1)][:,1], label="not zero")
plt.legend()
plt.xlim(0,1)
plt.ylim(0,1)
plt.show



acc_log = []
params_log = []
# evaluate the performance with random parameters for the kernel
## choose random params for the kernel
for i in range(1):
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

params = params_log[np.argmin(np.asarray(acc_log))]



plotting =  False
if plotting:
    precision = 20 # higher is preciser and more compute time

    # create a dummy dataset that uniformly spans the input space
    X_dummy = []
    for i in range(0,precision+1):
        for j in range(0,precision+1):
            X_dummy.append([i/precision,j/precision])
    X_dummy = np.asarray(X_dummy)
    print(len(X_dummy))

    # predict (about a minute on my laptop)
    y_dummy = svm_untrained_kernel.predict(X_dummy)

    # plot in order to observe the decision boundary
    plt.scatter(X_dummy[np.where(y_dummy == 1)[0],0], X_dummy[np.where(y_dummy == 1)[0],1], color="b", marker=".",label="dummy, 1")
    plt.scatter(X_dummy[np.where(y_dummy == -1)[0],0], X_dummy[np.where(y_dummy == -1)[0],1], color="r", marker=".",label="dummy, -1")
    plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker="+", label="train, 1")
    plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker="+", label="train, -1")
    plt.scatter(X_test[np.where(y_train == 1)[0],0], X_test[np.where(y_train == 1)[0],1], color="b", marker="x", label="test, 1")
    plt.scatter(X_test[np.where(y_train == -1)[0],0], X_test[np.where(y_train == -1)[0],1], color="r", marker="x", label="test, -1")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend()


# evaluate the performance with trained parameters for the kernel
## train the kernel
opt = qml.GradientDescentOptimizer(2)
for i in range(1000):
    subset = np.random.choice(list(range(len(X_train))), 4)
    params = opt.step(lambda _params: -k.target_alignment(X_train[subset], y_train[subset], _params), params)
    
    if (i+1) % 50 == 0:
        print("Step {} - Alignment on train = {:.3f}".format(i+1, k.target_alignment(X_train, y_train, params)))
## fit the SVM on the train set
svm_trained_kernel = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, params)).fit(X_train, y_train)
## evaluate the accuracy on the test set
trained_accuracy = accuracy(svm_trained_kernel, X_test, y_test)
print("with kernel training accuracy on test", trained_accuracy)
plotting = True
if plotting:
    precision = 20 # higher is preciser and more compute time

    # create a dummy dataset that uniformly spans the input space
    X_dummy = []
    for i in range(0,precision+1):
        for j in range(0,precision+1):
            X_dummy.append([i/precision,j/precision])
    X_dummy = np.asarray(X_dummy)
    print(len(X_dummy))

    # predict (about a minute on my laptop)
    y_dummy = svm_untrained_kernel.predict(X_dummy)

    # plot in order to observe the decision boundary
    plt.scatter(X_dummy[np.where(y_dummy == 1)[0],0], X_dummy[np.where(y_dummy == 1)[0],1], color="b", marker=".",label="dummy, 1")
    plt.scatter(X_dummy[np.where(y_dummy == -1)[0],0], X_dummy[np.where(y_dummy == -1)[0],1], color="r", marker=".",label="dummy, -1")
    plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker="+", label="train, 1")
    plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker="+", label="train, -1")
    plt.scatter(X_test[np.where(y_train == 1)[0],0], X_test[np.where(y_train == 1)[0],1], color="b", marker="x", label="test, 1")
    plt.scatter(X_test[np.where(y_train == -1)[0],0], X_test[np.where(y_train == -1)[0],1], color="r", marker="x", label="test, -1")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend()
# filename = "dataset_checkerboard.npy"
# with open(filename, 'wb') as f:
#    np.save(f, X_dummy)
#    np.save(f, y_dummy)
#    np.save(f, X_train)
#    np.save(f, y_train)
#    np.save(f, X_test)
#    np.save(f, y_test)
# +
#with open(filename, 'rb') as f:
#    X_dummy_c = np.load(f)
#    y_dummy_c = np.load(f)
#    X_train_c = np.load(f)
#    y_train_c = np.load(f)
#    X_test_c = np.load(f)
#    y_test_c = np.load(f)
# -



































































































































