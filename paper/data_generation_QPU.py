# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
import pennylane as qml
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets

import numpy as pure_np
from pennylane import numpy as np


import os 
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from wp_TK import tk_lib as tk

import matplotlib.pyplot as plt

import tqdm
from itertools import product

from pennylane_cirq import ops as cirq_ops
import nk_lib
from collections.abc import Iterable
import seaborn as sns

np.random.seed(42)
# -

# #  Gobal variables (used below)

features = 2
wires = [0,1,2]
depth = 3


# # Define dataset

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
    while (i<n_part):
        x1 = np.random.uniform(-.707,.707) # 0.707... = 0.5*\sqrt(2)
        x2 = np.random.uniform(-.707,.707)
        if((x1)*(x1) + x2*x2 < .5):
            i+=1
            X.append([1+x1,x2])
            if(x1*x1 + x2*x2 < .25):
                y.append(1)
            else:
                y.append(-1)
    
    i=0
    while(i<n_part):
        x1 = np.random.uniform(-.707,.707)
        x2 = np.random.uniform(-.707,.707)
        if(x1*x1 + x2*x2 <.5):
            i+=1
            X.append([x1-1,x2])
            if(x1*x1 + x2*x2 < .25):
                y.append(-1)
            else:
                y.append(1)
    
    i = 0
    while (i<n_test):
        x1 = np.random.uniform(-.707,.707)
        x2 = np.random.uniform(-.707,.707)
        if(x1*x1 + x2*x2 < .5):
            i+=1
            X_.append([1+x1,x2])
            if(x1*x1 + x2*x2 < .25):
                y_.append(1)
            else:
                y_.append(-1)
    
    i=0
    while(i<n_test):
        x1 = np.random.uniform(-.707,.707)
        x2 = np.random.uniform(-.707,.707)
        if(x1*x1 + x2*x2 <.5):
            i+=1
            X_.append([x1-1,x2])
            if(x1*x1 + x2*x2 < .25):
                y_.append(-1)
            else:
                y_.append(1)
            
    return X,y, X_,y_


X_train ,y_train, X_test, y_test = datagen(60,60)

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

# # Pretrained optimal parameters

filename = "data/parameters_symmetricdonuts_3_3.npy"
with open(filename, 'rb') as f:
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

# # Calculate upper triangle classical simulation for comparison

# +
def calculate_one_kernel_entry(embedding_kernel, datapoint_0, datapoint_1, params):    
    matrix_entry = embedding_kernel.kernel_matrix([datapoint_0], [datapoint_1], params)
    return(float(matrix_entry))


def calculate_upper_triangle_including_diagonal(embedding_kernel, datapoints, params):
    N = len(datapoints)
    matrix = [0] * N ** 2
    n_times = 0
    for i in range(N):
        for j in range(i, N):
            matrix[N * i + j] = calculate_one_kernel_entry(k, datapoints[i], datapoints[j], params)
            matrix[N * j + i] = matrix[N * i + j]
            n_times += 1
       
    return np.array(matrix).reshape((N, N))

dev = qml.device("default.qubit", wires=len(wires), shots=None)
k = qml.kernels.EmbeddingKernel(lambda x, params: ionq_ansatz(x, params, wires), dev)
square_kernel_matrix = calculate_upper_triangle_including_diagonal(k, X_train, params)


# +
def visualize_kernel_matrices(kernel_matrices, draw_last_cbar=False):
    num_mat = len(kernel_matrices)
    width_ratios = [1]*num_mat+[0.2]*int(draw_last_cbar)
    fig,ax = plt.subplots(1, num_mat+draw_last_cbar, figsize=(num_mat*5+draw_last_cbar, 5), gridspec_kw={'width_ratios': width_ratios})
    for i, kernel_matrix in enumerate(kernel_matrices):
        plot = sns.heatmap(
            kernel_matrix, 
            vmin=0,
            vmax=1,
            xticklabels='',
            yticklabels='',
            ax=ax,
            cmap='Spectral',
            cbar=True
        )
    if draw_last_cbar:
        ch = plot.get_children()
        fig.colorbar(ch[0], ax=ax[-2], cax=ax[-1])

square_kernel_matrix = np.load('testing_ionq_classical.npy')
 
visualize_kernel_matrices([square_kernel_matrix])
print(square_kernel_matrix)
# -

# # Simulation on ionq device

# +
shots_device = 175
# switch to amazon here:
bucket = "" # insert the name of the bucket
prefix = "" # insert the name of the folder in the bucket
s3_folder = (bucket, prefix)
dev_arn = "arn:aws:braket:::device/qpu/ionq/ionQdevice"
#print(safeguard)
# final safeguard: remove the comment
dev = qml.device("braket.aws.qubit", device_arn=dev_arn, s3_destination_folder=s3_folder, wires=len(wires), shots=shots_device, parallel=True)
k = qml.kernels.EmbeddingKernel(lambda x, params: ionq_ansatz(x, params, wires), dev)
ionq_device_kernel_matrix = calculate_upper_triangle_including_diagonal(k, X_train, params)

np.save('ionq_device_kernel_matrix', ionq_device_kernel_matrix)

