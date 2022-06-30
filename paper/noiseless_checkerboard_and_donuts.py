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

import pennylane as qml
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from tqdm.notebook import tqdm
from src.datasets import checkerboard, symmetric_donuts
from pickle import load, dump

# ## Choose your dataset and the recomputation options here. 

dataset = 'symmetric_donuts'
# dataset = 'symmetric_donuts'
# Activate the following to load the dataset instead of regenerating it. Note that the seed is fixed, so that
# there is no difference between the two options unless you change the seed.
load_dataset_from_file = True
# Activate the following to load the untrained results instead of recomputing them. Due to the seed for the random initial
# parameters being fixed, there is no difference unless you change the seed. Recomputing takes significant time.
load_results_from_file = True
# Activate the following to create a preliminary decision boundary plot. This requires additional resources
# if the data is not found in `dataset_filename`. See `noiseless_plot.py` for a proper decision boundary plot.
plot_decision_boundary = True

# +
samples_for_training = 4
if dataset=='checkerboard':
    np.random.seed(43) # Seed, 43 was used for the paper results
    num_wires = 5 # Number of qubits / Width of the circuit
    num_layers = 8 # Number of building blocks / "Depth" of the circuit
    num_train = 60 # Number of training datapoints
    num_test = 60 # Number of test datapoints
    num_random_params = 3 # Number of random parameter positions.
    num_epochs = 1000 # Number of epochs for target alignment training
    filename = 'data/checkerboard.pickle' # File to store the data in
    xlims = [0, 1]
    ylims = [0, 1]
    
elif dataset=='symmetric_donuts':
    np.random.seed(42) # Seed, 42 was used for the paper results
    num_wires = 3 # Number of qubits / Width of the circuit
    num_layers = 3 # Number of building blocks / "Depth" of the circuit
    num_train = 60 # Number of training datapoints
    num_test = 60 # Number of test datapoints
    num_random_params = 5 # Number of random parameter positions.
    num_epochs = 1000 # Number of epochs for target alignment training
    filename = 'data/symmetric_donuts.pickle' # File to store the data in
    xlims = [-2, 2]
    ylims = [-1, 1]

print(filename)


# +
# the Ansatz
def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding Ansatz.
    Args:
        x (ndarray): Data to be embedded.
        params (ndarray): Trainable circuit parameters.
        wires (qml.Wires): Qubits to act on.
        i0 (int): Wire on which the first embedding rotation gate acts.
        inc (int): Increment between wires on which the embedding rotation gates act.
    Comments:
        Note that this circuit iterates cyclically over the feature vector to be embedded.
    """
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])
        
    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

def ansatz(x, params, wires):
    """The embedding Ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))
# +
# get accuracy of a classifier against a target of labels.
def prediction_accuracy(y_predicted, y_target):
    return 1 - np.count_nonzero(y_predicted - y_target) / len(y_target)

# get random parameters for given QEK hyperparameters, uniformly distributed across [0, 2pi]
def get_random_params(num_wires, num_layers):
    return np.random.uniform(0, 2*np.pi, (num_layers, 2, num_wires))

stored_objects = [
    'X_plot',
    'y_plot_random',
    'y_plot_random_real',
    'y_plot_trained',
    'y_plot_trained_real',
    'X_train',
    'y_train',
    'X_test',
    'y_test',
    'random_params',
    'random_acc',
    'random_test_predict',
    'random_train_predict',
    'trained_params',
    'training_alignment',
    'trained_train_predict',
    'trained_test_predict',
]

# load from numpy storage but numbered
def load_data(obj, filename):
    with open(filename, 'rb') as f:
        _data = load(f)
    return _data[obj]


# +
# define the device
dev = qml.device("default.qubit", wires=num_wires)
wires = list(range(num_wires))

# init the embedding kernel
@qml.qnode(dev)
def kernel(x1, x2, params):
    ansatz(x1, params, wires)
    qml.adjoint(ansatz)(x2, params, wires)
    return qml.expval(qml.Projector([0]*num_wires, wires=wires))


# +
# Get the dataset, either by generating it (note the fixed seed) or by loading it from file
if load_dataset_from_file:
    load_objects = ['X_train', 'y_train', 'X_test', 'y_test']
    for obj in load_objects:
        exec(f"{obj} = load_data('{obj}', '{filename}')")
else:
    if dataset=='checkerboard':
        X_train, y_train, X_test, y_test = checkerboard(num_train, num_test, num_grid_row=4, num_grid_col=4)
    elif dataset=='symmetric_donuts':
        X_train, y_train, X_test, y_test = symmetric_donuts(num_train, num_test)

X_train_pos = X_train[np.where(y_train == 1)[0],:]
X_train_neg = X_train[np.where(y_train == -1)[0],:]
X_test_pos = X_test[np.where(y_test == 1)[0],:]
X_test_neg = X_test[np.where(y_test == -1)[0],:]
plt.scatter(X_train_pos[:,0], X_train_pos[:,1], color="b", marker=".", label="train, 1")
plt.scatter(X_train_neg[:,0], X_train_neg[:,1], color="r", marker=".", label="train, -1")
plt.scatter(X_test_pos[:,0], X_test_pos[:,1], color="b", marker="x", label="test, 1")
plt.scatter(X_test_neg[:,0], X_test_neg[:,1], color="r", marker="x", label="test, -1")
plt.xlim(xlims)
plt.ylim(ylims)
plt.legend()

# +
# evaluate the performance with random parameters for the kernel

if load_results_from_file:
    load_objects = ['random_params', 'random_acc', 'random_train_predict', 'random_test_predict']
    for obj in load_objects:
        exec(f"{obj} = load_data('{obj}', '{filename}')")
    for i in range(num_random_params):
        untrained_accuracy_test = prediction_accuracy(random_test_predict[i], y_test)
        print("without kernel training accuracy", untrained_accuracy_test)
else:
    random_acc = []
    random_params = []
    random_train_predict = []
    random_test_predict = []
    ## choose random params for the kernel
    for i in tqdm(range(num_random_params)):
        params = get_random_params(num_wires, num_layers)
        ## fit the SVM on the training data
        mapped_kernel = lambda X1, X2: kernel(X1, X2, params)
        mapped_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, mapped_kernel)
        svm_untrained_kernel = SVC(kernel=mapped_kernel_matrix).fit(X_train, y_train)
        ## evaluate on the test set
#         untrained_accuracy = accuracy(svm_untrained_kernel, X_test, y_test)
        random_train_predict.append(svm_untrained_kernel.predict(X_train))
        random_test_predict.append(svm_untrained_kernel.predict(X_test))
        untrained_accuracy_train = prediction_accuracy(random_train_predict[-1], y_train)
        untrained_accuracy_test = prediction_accuracy(random_test_predict[-1], y_test)
        print("without kernel training accuracy", untrained_accuracy_test)
        random_acc.append(untrained_accuracy_test)
        random_params.append(params)
# -

min_accuracy = np.min(random_acc)
min_params = np.array(random_params[np.argmin(random_acc)], requires_grad=True)
print("Untrained accuracies:", random_acc)
print(f"We will train starting with the minimal accuracy {min_accuracy} at parameters\n{min_params}")


if plot_decision_boundary:
    try:
        load_objects = ['X_plot', 'y_plot_random_real', 'y_plot_random']
        for obj in load_objects:
            exec(f"{obj} = load_data('{obj}', '{filename}')")
    except:
        # create a dummy dataset that uniformly spans the input space for decision boundary plotting
        precision = 20 # higher is more precise and more compute time
        X_plot = []
        for i in range(0,precision+1):
            for j in range(0,precision+1):
                X_plot.append([i/precision,j/precision])
        X_plot = np.asarray(X_plot)
        # predict
        y_plot_random_real = svm_untrained_kernel.decision_function(X_plot)
        y_plot_random = np.sign(y_plot_random_real)
    X_plot_pos = X_plot[np.where(y_plot_random == 1)[0],:]
    X_plot_neg = X_plot[np.where(y_plot_random == -1)[0],:]
    
    # plot in order to observe the decision boundary
    plt.scatter(X_plot_pos[:,0], X_plot_pos[:,1], color="b", marker=".")
    plt.scatter(X_plot_neg[:,0], X_plot_neg[:,1], color="r", marker=".")
    # plot training and test data
    plt.scatter(X_train_pos[:,0], X_train_pos[:,1], color="b", marker="+", label="train, 1")
    plt.scatter(X_train_neg[:,0], X_train_neg[:,1], color="r", marker="+", label="train, -1")
    plt.scatter(X_test_pos[:,0], X_test_pos[:,1], color="b", marker="x", label="test, 1")
    plt.scatter(X_test_neg[:,0], X_test_neg[:,1], color="r", marker="x", label="test, -1")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend(bbox_to_anchor=(1.01,1.))


"""
Currently, the function ``qml.kernels.target_alignment`` is not
differentiable yet, making it unfit for gradient descent optimization.
We therefore first define a differentiable version of this function.
"""
def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """Kernel-target alignment between kernel and labels."""

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm

    return inner_product


# evaluate the performance with trained parameters for the kernel
if load_results_from_file:
    load_objects = ['y_plot_trained', 'y_plot_trained_real', 'trained_params', 'training_alignment',
                    'trained_train_predict', 'trained_test_predict']
    for obj in load_objects:
        exec(f"{obj} = load_data('{obj}', '{filename}')")
    trained_train_accuracy = prediction_accuracy(trained_train_predict, y_train)
    trained_test_accuracy = prediction_accuracy(trained_test_predict, y_test)
    print("with kernel training accuracy on test", trained_test_accuracy)
else:
    ## train the kernel
    training_alignment = []
    mapped_kernel = lambda X1, X2: kernel(X1, X2, params)
    alignment = target_alignment(
        X_train,
        y_train,
        mapped_kernel,
        assume_normalized_kernel=True,
        rescale_class_labels=True,
    )
    training_alignment.append(alignment)
    print("Step 0 - Alignment on train = {:.3f}".format(alignment))
    opt = qml.GradientDescentOptimizer(2)
    for i in tqdm(range(num_epochs)):
        # We train on a small random subset of the training data set
        subset = np.random.choice(list(range(len(X_train))), 4)
        mapped_neg_alignment = lambda par: -target_alignment(
            X_train[subset],
            y_train[subset],
            lambda X1, X2: kernel(X1, X2, par),
            assume_normalized_kernel=True,
            rescale_class_labels=True,
        )
        params = opt.step(mapped_neg_alignment, params)
        if (i+1) % 50 == 0:
            mapped_kernel = lambda X1, X2: kernel(X1, X2, params)
            alignment = target_alignment(
                X_train,
                y_train,
                mapped_kernel,
                assume_normalized_kernel=True,
                rescale_class_labels=True,
            )
            training_alignment.append(alignment)
            print("Step {} - Alignment on train = {:.3f}".format(i+1, alignment))
    trained_params = params.copy()
    ## fit the SVM on the train set
    mapped_kernel = lambda X1, X2: kernel(X1, X2, trained_params)
    mapped_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, mapped_kernel)
    svm_trained_kernel = SVC(kernel=mapped_kernel_matrix).fit(X_train, y_train)
    ## evaluate the accuracy on the test set
    trained_train_predict = svm_trained_kernel.predict(X_train) 
    trained_test_predict = svm_trained_kernel.predict(X_test)
    trained_train_accuracy = prediction_accuracy(trained_train_predict, y_train)
    trained_test_accuracy = prediction_accuracy(trained_test_predict, y_test)
    print("with kernel training accuracy on test", trained_test_accuracy)
    # predict (about a minute on my laptop)
    y_plot_trained_real = svm_trained_kernel.decision_function(X_plot)
    y_plot_trained = np.sign(y_plot_trained_real)

# +
# Store data - only activate if you are really sure about what you are doing.
# _data = {obj: globals()[obj] for obj in stored_objects}
# with open(filename, 'wb') as f:
#     dump(_data, f)
# -







