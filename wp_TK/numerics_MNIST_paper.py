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

np.random.seed(23)


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
width = 4
depth = 32 


dev = qml.device("default.qubit", wires=width)
wires = list(range(width))

# init the embedding kernel
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x, params, wires), dev)
# +
# load the dataset and split

(train_X, train_y), (test_X, test_y) = mnist.load_data()

sample_size = 1000 # leave high here, change in next step

train_idx0 = np.argwhere(train_y == 0)[:sample_size]
train_X0 = train_X[train_idx0].squeeze() * np.pi/255

train_idx1 = np.argwhere(train_y == 1)[:sample_size]
train_X1 = train_X[train_idx1].squeeze() * np.pi/255

#X = np.vstack([train_X0, train_X1])
#y = np.hstack([[-1]*sample_size, [1]*sample_size])

X_train = np.vstack([train_X0[:sample_size], train_X1[:sample_size]])
y_train = np.hstack([[-1]*sample_size, [1]*sample_size])

test_idx0 = np.argwhere(test_y == 0)[:sample_size]
test_X0 = test_X[test_idx0].squeeze() * np.pi/255

test_idx1 = np.argwhere(test_y == 1)[:sample_size]
test_X1 = test_X[test_idx1].squeeze() * np.pi/255

X_test = np.vstack([test_X0[:sample_size], test_X1[:sample_size]])
y_test = np.hstack([[-1]*sample_size, [1]*sample_size])

# -



# +
# visual check
gs = mpl.gridspec.GridSpec(2, 10) 
fig = plt.figure(figsize=(8,2))

print(y_train)
for j in range(10):
    ax=plt.subplot(gs[0, j])
    plt.imshow(X_train[j], cmap=plt.get_cmap('gray'))
    ax.axis("off")
    
    ax=plt.subplot(gs[1, j])
    plt.imshow(X_train[5+j], cmap=plt.get_cmap('gray'))
    ax.axis("off")    
plt.show()

print(y_test)
gs = mpl.gridspec.GridSpec(2, 10) 
fig = plt.figure(figsize=(8,2))
for j in range(10):
    ax=plt.subplot(gs[0, j])
    plt.imshow(X_test[j], cmap=plt.get_cmap('gray'))
    ax.axis("off")
    
    ax=plt.subplot(gs[1, j])
    plt.imshow(X_test[5+j], cmap=plt.get_cmap('gray'))
    ax.axis("off")  
plt.show()
# -























































# +
# Process images with y=0
samples_per_image = 3
## threshold
x_0, x_1 = np.where(train_X0[0]>=0.95)
## sample $size datapoints from this image
sample_indices = np.random.randint(low=0, high=len(x_0), size=samples_per_image)
## add the coordinates to our dataset
zero_samples = [x_0[sample_indices], x_1[sample_indices]]

for i in range(1,500):
    ## threshold
    x_0, x_1 = np.where(train_X0[i]>=0.95)
    ## sample $size datapoints from this image
    sample_indices = np.random.randint(low=0, high=len(x_0), size=samples_per_image)
    ## add the coordinates to out dataset
    zero_samples[0] = np.hstack([zero_samples[0], x_0[sample_indices]])
    zero_samples[1] = np.hstack([zero_samples[1], x_1[sample_indices]])

## visual check
plt.scatter(zero_samples[0], zero_samples[1])
plt.xlim(0,28)
plt.ylim(0,28)
plt.show()

# Process images with y=1
## threshold
x_0, x_1 = np.where(train_X1[0]>=0.95)
## sample $size datapoints from this image
sample_indices = np.random.randint(low=0, high=len(x_0), size=samples_per_image)
## add the coordinates to our dataset
one_samples = [x_0[sample_indices], x_1[sample_indices]]

for i in range(2,1000):
    ## threshold
    x_0, x_1 = np.where(train_X1[i]>=0.95)
    ## sample $size datapoints from this image
    sample_indices = np.random.randint(low=0, high=len(x_0), size=samples_per_image)
    ## add the coordinates to our dataset
    one_samples[0] = np.hstack([one_samples[0], x_0[sample_indices]])
    one_samples[1] = np.hstack([one_samples[1], x_1[sample_indices]])

# visual check    
plt.scatter(one_samples[0], one_samples[1])
plt.xlim(0,28)
plt.ylim(0,28)
plt.show()



# +
# split the two datasets, containing y=0 and y=1 respectively, into 5 datasets:
## distinct_zeros: samples unique to y=0
## distinct_ones: samples unique to y=1
## duplicates: samples that are both in y=0 and y=1
## not_ones: samples that are not in the "unique to y=0" set
## not_zeros: samples that are not in the "unique to y=1" set

zeros = np.unique(np.asarray(zero_samples)[:].T, axis=0)
ones = np.unique(np.asarray(one_samples)[:].T, axis=0)
distinct_zeros = []
distinct_ones = []
duplicates = []
# find unique zeros and duplicates
for sample in zeros:
    first_index = np.where(ones[:,0] == sample[0])
    if(len(first_index)>0):
        second_index = np.where(ones[first_index][:,1] == sample[1])
        if(len(second_index[0])>0):
            duplicates.append(sample)
        else:
            distinct_zeros.append(sample)
    else:
        distinct_zeros.append(sample)

# find unique ones: 
for sample in ones:
    first_index = np.where(zeros[:,0] == sample[0])
    if(len(first_index)>0):
        second_index = np.where(zeros[first_index][:,1] == sample[1])
        if(len(second_index[0])>0):
            continue
        else:
            distinct_ones.append(sample)
    else:
        distinct_ones.append(sample)
        
# normalize        
distinct_zeros = np.asarray(distinct_zeros).T/28
duplicates = np.asarray(duplicates).T/28
distinct_ones = np.asarray(distinct_ones).T/28

not_zeros = np.hstack([distinct_ones, duplicates])
not_ones = np.hstack([distinct_zeros, duplicates])

# visual inspection

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

plt.title("Not zeros")
plt.scatter(not_zeros[0], not_zeros[1])
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

plt.title("Not ones")
plt.scatter(not_ones[0], not_ones[1])
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
# -
len(distinct_ones[0])






















# # Zero vs not-zero




# +
# split the dataset in train and test by means of array indices
## create the train indices
train_zero_indices = np.random.randint(low=0, high=len(distinct_zeros[0]), size=int(0.5*len(distinct_zeros[0])))
np.random.shuffle(train_zero_indices)
train_not_zero_indices = np.random.randint(low=0, high=len(not_zeros[0]), size=int(0.5*len(distinct_zeros[0])))
np.random.shuffle(train_not_zero_indices)

# create the test indices
all_indices = range(len(distinct_zeros[0]))
test_zero_indices = []
for x in all_indices:
    if x not in train_zero_indices:
        test_zero_indices.append(x)
test_zero_indices = np.asarray(test_zero_indices)
np.random.shuffle(test_zero_indices)

all_indices = range(len(not_zeros[0]))
test_not_zero_indices = []
for x in all_indices:
    if x not in test_not_zero_indices:
        test_not_zero_indices.append(x)
test_not_zero_indices = np.asarray(test_not_zero_indices)
np.random.shuffle(test_zero_indices)
# -



# select $sample_size "zero" and $sample_size "non-zero" samples as training data 
sample_size = 15
X_train = np.vstack([distinct_zeros.T[train_zero_indices][:sample_size], not_zeros.T[train_not_zero_indices][:sample_size]])
y_train = np.hstack([np.ones((sample_size)), -np.ones((sample_size))])
# and test data
X_test = np.vstack([distinct_zeros.T[test_zero_indices][:sample_size], not_zeros.T[test_not_zero_indices][:sample_size]])
y_test = np.hstack([np.ones((sample_size)), -np.ones((sample_size))])














plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker="+", label="train, zero")
plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker="+", label="train, not-zero")
plt.scatter(X_test[np.where(y_test == 1)[0],0], X_test[np.where(y_test == 1)[0],1], color="b", marker="x", label="test, zero")
plt.scatter(X_test[np.where(y_test == -1)[0],0], X_test[np.where(y_test == -1)[0],1], color="r", marker="x", label="test, not-zero")
plt.legend()
plt.show()

acc_log = []
params_log = []
# evaluate the performance with random parameters for the kernel
## choose random params for the kernel
for i in range(3):
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



plotting =  True
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
    plt.scatter(X_dummy[np.where(y_dummy == 1)[0],0], X_dummy[np.where(y_dummy == 1)[0],1], color="b", marker=".",label="grid, zero")
    plt.scatter(X_dummy[np.where(y_dummy == -1)[0],0], X_dummy[np.where(y_dummy == -1)[0],1], color="r", marker=".",label="grid, not zero")
    plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker="+", label="train, zero")
    plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker="+", label="train, not zero")
    plt.scatter(X_test[np.where(y_train == 1)[0],0], X_test[np.where(y_train == 1)[0],1], color="b", marker="x", label="test, zero")
    plt.scatter(X_test[np.where(y_train == -1)[0],0], X_test[np.where(y_train == -1)[0],1], color="r", marker="x", label="test, not zero")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend()


# evaluate the performance with trained parameters for the kernel
## train the kernel
opt = qml.GradientDescentOptimizer(2)
for i in range(500):
    subset = np.random.choice(list(range(len(X_train))), 4)
    params = opt.step(lambda _params: -k.target_alignment(X_train[subset], y_train[subset], _params), params)
    if (i+1) % 50 == 0:
        print("Step {} - Alignment on train = {:.3f}".format(i+1, k.target_alignment(X_train, y_train, params)))
opt = qml.GradientDescentOptimizer(1)
for i in range(500):
    subset = np.random.choice(list(range(len(X_train))), 4)
    params = opt.step(lambda _params: -k.target_alignment(X_train[subset], y_train[subset], _params), params)
    if (i+1) % 50 == 0:
        print("Step {} - Alignment on train = {:.3f}".format(i+1, k.target_alignment(X_train, y_train, params)))
opt = qml.GradientDescentOptimizer(0.5)
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
    plt.scatter(X_dummy[np.where(y_dummy == 1)[0],0], X_dummy[np.where(y_dummy == 1)[0],1], color="b", marker=".",label="grid, zero")
    plt.scatter(X_dummy[np.where(y_dummy == -1)[0],0], X_dummy[np.where(y_dummy == -1)[0],1], color="r", marker=".",label="grid, not zero")
    plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker="+", label="train, zero")
    plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker="+", label="train, not zero")
    plt.scatter(X_test[np.where(y_test == 1)[0],0], X_test[np.where(y_test == 1)[0],1], color="b", marker="x", label="test, zero")
    plt.scatter(X_test[np.where(y_test == -1)[0],0], X_test[np.where(y_test == -1)[0],1], color="r", marker="x", label="test, not zero")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend()
# # One vs not-one
# +
# split the dataset in train and test by means of array indices
## create the train indices
train_one_indices = np.random.randint(low=0, high=len(distinct_ones[0]), size=int(0.5*len(distinct_ones[0])))
np.random.shuffle(train_one_indices)
train_not_one_indices = np.random.randint(low=0, high=len(not_ones[0]), size=int(0.5*len(distinct_ones[0])))
np.random.shuffle(train_not_one_indices)

# create the test indices
all_indices = range(len(distinct_ones[0]))
test_one_indices = []
for x in all_indices:
    if x not in train_one_indices:
        test_one_indices.append(x)
test_one_indices = np.asarray(test_one_indices)
np.random.shuffle(test_one_indices)

all_indices = range(len(not_ones[0]))
test_not_one_indices = []
for x in all_indices:
    if x not in test_not_one_indices:
        test_not_one_indices.append(x)
test_not_one_indices = np.asarray(test_not_one_indices)
np.random.shuffle(test_one_indices)
# -


len(distinct_ones[0])


# select $sample_size "one" and $sample_size "non-one" samples as training data 
sample_size = 15 # less data
X_train_1 = np.vstack([distinct_ones.T[train_one_indices][:sample_size], not_ones.T[train_not_one_indices][:sample_size]])
y_train_1 = np.hstack([np.ones((sample_size)), -np.ones((sample_size))])
# and test data
X_test_1 = np.vstack([distinct_ones.T[test_one_indices][:sample_size], not_ones.T[test_not_one_indices][:sample_size]])
y_test_1 = np.hstack([np.ones((sample_size)), -np.ones((sample_size))])


plt.scatter(X_train_1[np.where(y_train_1 == 1)[0],0], X_train_1[np.where(y_train_1 == 1)[0],1], color="b", marker="+", label="train, one")
plt.scatter(X_train_1[np.where(y_train_1 == -1)[0],0], X_train_1[np.where(y_train_1 == -1)[0],1], color="r", marker="+", label="train, not one")
plt.scatter(X_test_1[np.where(y_test_1 == 1)[0],0], X_test_1[np.where(y_test_1 == 1)[0],1], color="b", marker="x", label="test, one")
plt.scatter(X_test_1[np.where(y_test_1 == -1)[0],0], X_test_1[np.where(y_test_1 == -1)[0],1], color="r", marker="x", label="test, not one")
plt.legend()
plt.show()

acc_log_1 = []
params_log_1 = []
# evaluate the performance with random parameters for the kernel
## choose random params for the kernel
for i in range(3):
    params_1 = random_params(width, depth)
    #print(params)
    ## fit the SVM on the training data
    svm_untrained_kernel_1 = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, params_1)).fit(X_train_1, y_train_1)
    ## evaluate on the test set
    untrained_accuracy_1 = accuracy(svm_untrained_kernel_1, X_test_1, y_test_1)
    print("without kernel training accuracy", untrained_accuracy_1)
    acc_log_1.append(untrained_accuracy_1)
    params_log_1.append(params_1)
print("going with", acc_log_1[np.argmin(np.asarray(acc_log_1))])
params_1 = params_log_1[np.argmin(np.asarray(acc_log_1))]

print("Untrained accuracies:", acc_log_1)

params_1 = params_log_1[np.argmin(np.asarray(acc_log_1))]

plotting =  True
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
    y_dummy = svm_untrained_kernel_1.predict(X_dummy)

    # plot in order to observe the decision boundary
    plt.scatter(X_dummy[np.where(y_dummy == 1)[0],0], X_dummy[np.where(y_dummy == 1)[0],1], color="b", marker=".",label="grid, one")
    plt.scatter(X_dummy[np.where(y_dummy == -1)[0],0], X_dummy[np.where(y_dummy == -1)[0],1], color="r", marker=".",label="grid, not one")
    plt.scatter(X_train_1[np.where(y_train_1 == 1)[0],0], X_train_1[np.where(y_train_1 == 1)[0],1], color="b", marker="+", label="train, one")
    plt.scatter(X_train_1[np.where(y_train_1 == -1)[0],0], X_train_1[np.where(y_train_1 == -1)[0],1], color="r", marker="+", label="train, not one")
    plt.scatter(X_test_1[np.where(y_train_1 == 1)[0],0], X_test_1[np.where(y_train_1 == 1)[0],1], color="b", marker="x", label="test, one")
    plt.scatter(X_test_1[np.where(y_train_1 == -1)[0],0], X_test_1[np.where(y_train_1 == -1)[0],1], color="r", marker="x", label="test, not one")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend()

# evaluate the performance with trained parameters for the kernel
## train the kernel
opt = qml.GradientDescentOptimizer(2)
for i in range(500):
    subset = np.random.choice(list(range(len(X_train_1))), 4)
    params_1 = opt.step(lambda _params_1: -k.target_alignment(X_train_1[subset], y_train_1[subset], _params_1), params_1)  
    if (i+1) % 50 == 0:
        print("Step {} - Alignment on train = {:.3f}".format(i+1, k.target_alignment(X_train_1, y_train_1, params_1)))
opt = qml.GradientDescentOptimizer(1)
for i in range(500):
    subset = np.random.choice(list(range(len(X_train_1))), 4)
    params_1 = opt.step(lambda _params_1: -k.target_alignment(X_train_1[subset], y_train_1[subset], _params_1), params_1)  
    if (i+1) % 50 == 0:
        print("Step {} - Alignment on train = {:.3f}".format(i+1, k.target_alignment(X_train_1, y_train_1, params_1)))
opt = qml.GradientDescentOptimizer(0.5)
for i in range(1000):
    subset = np.random.choice(list(range(len(X_train_1))), 4)
    params_1 = opt.step(lambda _params_1: -k.target_alignment(X_train_1[subset], y_train_1[subset], _params_1), params_1)
    
    if (i+1) % 50 == 0:
        print("Step {} - Alignment on train = {:.3f}".format(i+1, k.target_alignment(X_train_1, y_train_1, params_1)))
## fit the SVM on the train set
svm_trained_kernel_1 = SVC(kernel=lambda X1, X2: k.kernel_matrix(X1, X2, params_1)).fit(X_train_1, y_train_1)
## evaluate the accuracy on the test set
trained_accuracy_1 = accuracy(svm_trained_kernel_1, X_test_1, y_test_1)
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
    y_dummy = svm_untrained_kernel_1.predict(X_dummy)

    # plot in order to observe the decision boundary
    plt.scatter(X_dummy[np.where(y_dummy == 1)[0],0], X_dummy[np.where(y_dummy == 1)[0],1], color="b", marker=".",label="grid, one")
    plt.scatter(X_dummy[np.where(y_dummy == -1)[0],0], X_dummy[np.where(y_dummy == -1)[0],1], color="r", marker=".",label="grid, not one")
    plt.scatter(X_train_1[np.where(y_train_1 == 1)[0],0], X_train_1[np.where(y_train_1 == 1)[0],1], color="b", marker="+", label="train, one")
    plt.scatter(X_train_1[np.where(y_train_1 == -1)[0],0], X_train_1[np.where(y_train_1 == -1)[0],1], color="r", marker="+", label="train, not one")
    plt.scatter(X_test_1[np.where(y_test_1 == 1)[0],0], X_test_1[np.where(y_test_1 == 1)[0],1], color="b", marker="x", label="test, one")
    plt.scatter(X_test_1[np.where(y_test_1 == -1)[0],0], X_test_1[np.where(y_test_1 == -1)[0],1], color="r", marker="x", label="test, not one")
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
# testing images that are zero
sample_size = 15
for i in range(5):
    current_sample = i
    #current_test_image = test_X0[current_sample]
    x_0, x_1 = np.asarray(np.where(test_X0[current_sample]>=0.95))/28
    x = np.asarray([x_0, x_1])
    certainty = 0.
    iterations = 0
    while (certainty < 0.11 and iterations <5):
        print("Iteration:", iterations)
        test_indices = np.random.randint(low=0, high=len(x.T), size=sample_size)
        samples = x.T[test_indices]

        plt.scatter(x_0, x_1)
        plt.scatter(samples[:,0], samples[:,1])
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.show()

        zero_or_not = svm_trained_kernel.predict(samples) # zero or not
        one_or_not = svm_trained_kernel_1.predict(samples) # zero or not
        certainty = np.absolute(((np.sum(zero_or_not)+15)/30)-((np.sum(one_or_not)+15)/30))
        iterations += 1
        print("Certainty:", (np.sum(zero_or_not)+15)/30, (np.sum(one_or_not)+15)/30, certainty)
        print("Classification:", np.argmax([np.sum(zero_or_not), np.sum(one_or_not)]))
    
sample_size = 15
for i in range(5):
    current_sample = i
    #current_test_image = test_X1[current_sample]
    x_0, x_1 = np.asarray(np.where(test_X1[current_sample]>=0.95))/28
    x = np.asarray([x_0, x_1])
    certainty = 0.
    iterations = 0
    while (certainty < 0.11 and iterations <5):
        print("Iteration:", iterations)
        test_indices = np.random.randint(low=0, high=len(x.T), size=sample_size)
        samples = x.T[test_indices]

        plt.scatter(x_0, x_1)
        plt.scatter(samples[:,0], samples[:,1])
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.show()

        zero_or_not = svm_trained_kernel.predict(samples) # zero or not
        one_or_not = svm_trained_kernel_1.predict(samples) # zero or not
        certainty = np.absolute(((np.sum(zero_or_not)+15)/30)-((np.sum(one_or_not)+15)/30))
        iterations += 1
        print("Certainty:", (np.sum(zero_or_not)+15)/30, (np.sum(one_or_not)+15)/30, certainty)
        print("Classification:", np.argmax([np.sum(zero_or_not), np.sum(one_or_not)]))
    
# -



a = -7
b = -11
certainty = (a+15)/30, (b+15)/30, np.absolute(((a+15)/30)-((b+15)/30))
print(certainty)

np.sum(zero_or_not)

np.sum(one_or_not)

len(np.where(zero_or_not==1)[0])

len(np.where(zero_or_not==-1)[0])

len(np.where(one_or_not==1)[0])

np.argmax([-7, -11]) 










filename = "dataset_MNIST_23.npy"
with open(filename, 'wb') as f:
    np.save(f, params_log[np.argmin(np.asarray(acc_log))]) # random params for "zero" vs "not-zero"
    np.save(f, params) # trained params for "zero" vs "not-zero"
    np.save(f, params_log_1[np.argmin(np.asarray(acc_log_1))]) # random params for "one" vs "not-one"
    np.save(f, params_1) # trained params for "one" vs "not-one"
    np.save(f, X_train)
    np.save(f, y_train)
    np.save(f, X_test)
    np.save(f, y_test)


params_log[np.argmin(np.asarray(acc_log))]

















































































































































