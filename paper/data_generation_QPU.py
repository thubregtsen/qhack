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

# # Computing the kernel matrix on a QPU
#
# Here we execute the QEK on a real QPU by IonQ. By default, the notebook instead uses a noiseless simulator.
# Using the simulator, the notebook takes a few minutes on a laptop computer.

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import noisy_helper_functions as nhf

# #  Gobal variables (used below)

np.random.seed(42)
features = 2
wires = [0,1,2]
depth = 3
# File containing pretrained QEK parameters
filename_opt_param = "data/parameters_symmetricdonuts_3_3.npy"


# # Define dataset

def datagen (n_train, n_test):
    """generate data in two circles, with flipped label regions
    Args:
        n_train (int): Number of train datapoints
        n_test (int): Number of test datapoints
    Returns:
        X_train (ndarray): Training datapoints
        y_train (ndarray): Training labels
        X_test (ndarray): Testing datapoints
        y_test (ndarray): Testing labels
    """
    # 
    # the radii are chosen so that data is balanced
    inv_sqrt2 = 1/np.sqrt(2)
    
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    # Generate the training dataset
    x_donut = 1
    i = 0
    while (i<n_train):
        x = np.random.uniform(-inv_sqrt2,inv_sqrt2, 2)
        r_squared = np.linalg.norm(x, 2)**2
        if r_squared < 0.5:
            i += 1
            X_train.append([x_donut+x[0],x[1]])
            if r_squared < .25:
                y_train.append(x_donut)
            else:
                y_train.append(-x_donut)
            # Move over to second donut 
            if i==n_train//2:
                x_donut = -1
    
    # Generate the testing dataset
    x_donut = 1
    i = 0
    while (i<n_test):
        x = np.random.uniform(-inv_sqrt2,inv_sqrt2, 2)
        r_squared = np.linalg.norm(x, 2)**2
        if r_squared < 0.5:
            i += 1
            X_test.append([x_donut+x[0],x[1]])
            if r_squared < .25:
                y_test.append(x_donut)
            else:
                y_test.append(-x_donut)
            # Move over to second donut 
            if i==n_test//2:
                x_donut = -1
            
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


X_train ,y_train, X_test, y_test = datagen(60, 60)

X_train.shape, X_test.shape

# plot for visual inspection
print("The train and test data is as follows:")
plt.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="b", marker=".", label="train, 1")
plt.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="r", marker=".", label="train, -1")
plt.scatter(X_test[np.where(y_test == 1)[0],0], X_test[np.where(y_test == 1)[0],1], color="b", marker="x", label="test, 1")
plt.scatter(X_test[np.where(y_test == -1)[0],0], X_test[np.where(y_test == -1)[0],1], color="r", marker="x", label="test, -1")
plt.ylim([-1, 1])
plt.xlim([-2, 2])
plt.legend()
plt.show()

# # Pretrained optimal parameters

with open(filename_opt_param, 'rb') as f:
    params= np.load(f)
print(params)


# # Circuit compiled to native gates of IonQ Device

# +
def ionq_layer(x, params, wires, i0=0, inc=1):
    i = i0
    N = len(wires)
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
    for j, wire in enumerate(wires):    
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
    for j, wire in enumerate(wires):
        qml.RY(params[0, j], wires=[wire])
        
    qml.CNOT(wires= wires[:2])
    qml.RZ(-params[1][0]/2, wires=1)
    qml.CNOT(wires= wires[:2])
    qml.RZ(params[1][0]/2, wires=1)
    
    qml.CNOT(wires=wires[1:3])
    qml.RZ(-params[1][1]/2, wires=2)
    qml.CNOT(wires= wires[1:3])
    qml.RZ(params[1][1]/2, wires=2)
        
    qml.CNOT(wires=[wires[2],wires[0]])
    qml.RZ(-params[1][2]/2, wires=0)
    qml.CNOT(wires= [wires[2],wires[0]])
    qml.RZ(params[1][2]/2, wires=0)


@qml.template
def ionq_ansatz(x, params, wires):
    for j, layer_params in enumerate(params):
        ionq_layer(x, layer_params, wires, i0=j * len(wires))


# -

# # Calculate kernel matrix via classical simulation

dev = qml.device("default.qubit", wires=len(wires), shots=175)
k = qml.kernels.EmbeddingKernel(lambda x, params: ionq_ansatz(x, params, wires), dev)
simulated_kernel_matrix = k.square_kernel_matrix(X_train, params)

nhf.visualize_kernel_matrices([simulated_kernel_matrix])
print(simulated_kernel_matrix)

# # Computation on ionq device

# +
shots_device = 175
try:
    # switch to amazon here:
    bucket = "" # insert the name of the bucket
    prefix = "" # insert the name of the folder in the bucket
    s3_folder = (bucket, prefix)
    dev_arn = "arn:aws:braket:::device/qpu/ionq/ionQdevice"
    dev = qml.device("braket.aws.qubit", device_arn=dev_arn, s3_destination_folder=s3_folder, wires=len(wires), shots=shots_device, parallel=True)
    k = qml.kernels.EmbeddingKernel(lambda x, params: ionq_ansatz(x, params, wires), dev)
    ionq_device_kernel_matrix = calculate_upper_triangle_including_diagonal(k, X_train, params)
    hardware_ran = True
except:
    ionq_device_kernel_matrix = simulated_kernel_matrix
    hardware_ran = False
    
nhf.visualize_kernel_matrices([ionq_device_kernel_matrix])
# -


