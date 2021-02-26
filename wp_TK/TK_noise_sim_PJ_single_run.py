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

# # Imports

import pennylane as qml
from pennylane import numpy as np
import tk_lib as tk
from cake_dataset import Dataset as Cake
from cake_dataset import DoubleCake
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
from sklearn.svm import SVC
import pandas as pd

# # Dataset with plot, prepare data for classification.

# +
dataset = DoubleCake(0, 6)
dataset.plot(plt.gca())

X = np.vstack([dataset.X, dataset.Y]).T
Y = dataset.labels_sym.astype(int)
# -

# # Random, zero and optimal parameters

rnd_param = np.tensor([[[3.83185811, 2.43217597, 6.04150259, 6.10219181, 2.24859771],
         [2.10229161, 3.01695202, 0.65963585, 3.01146847, 3.09878739]],

        [[2.98450446, 4.67620615, 2.65282874, 0.27375408, 3.51592262],
         [4.42306178, 2.10907678, 1.9859467 , 3.15253185, 5.1835622 ]],

        [[3.15053375, 1.15141625, 6.26411875, 1.4094818 , 2.89303727],
         [0.88448723, 1.37280759, 1.42852862, 2.79908337, 4.82479853]],

        [[2.96944762, 2.92050829, 5.08902411, 4.38583442, 4.57381108],
         [2.87380533, 2.79339977, 5.40042108, 1.22715656, 3.55334794]],

        [[4.85217317, 2.32865449, 3.36674732, 5.37284552, 4.41718962],
         [5.46919267, 4.1238232 , 5.63482497, 1.35359693, 1.55163904]],

        [[4.7955417 , 1.71132909, 3.45214701, 1.30618948, 2.43551656],
         [5.99802411, 0.86416771, 1.52129757, 4.48878166, 5.1649024 ]]], requires_grad=True)

zero_param = np.zeros_like(rnd_param)

opt_param = np.tensor([[[ 3.10041694,  6.28648541,  0.17216709,  3.41982814,
           5.43972889],
         [ 8.61896368,  2.96729878,  1.3804001 ,  3.63942291,
           5.53767498]],

        [[ 1.45857959,  3.50437136,  4.68830514,  5.94224399,
           1.46699806],
         [ 7.25085301,  0.34349336,  3.60122049,  3.76097482,
           1.43088235]],

        [[ 1.4319011 ,  1.09293963,  2.44024258,  4.63729544,
          -0.47946884],
         [ 0.39648489, -0.0749347 ,  3.13934162,  1.18520239,
          -0.68025965]],

        [[ 1.86128989,  1.08249672,  0.5714569 ,  4.16301649,
           2.80128062],
         [ 6.51381847,  5.65047927,  0.36280346,  5.26370351,
           6.03065534]],

        [[ 1.64724801,  2.49202953,  4.14132734,  2.40267736,
           7.01295776],
         [-0.53129869, -0.32494442,  4.69311076,  6.36346591,
           2.91400076]],

        [[ 5.41552666,  3.80159486,  1.96329725,  1.70261348,
           3.13386862],
         [ 0.45227504,  4.11186956,  5.91495654,  3.46714211,
           3.92814319]]], requires_grad=True)

W = np.random.normal(0, .7, (2, 30))

# # Evaluate a single circuit

# +
dataset = DoubleCake(0, 6)
dataset.plot(plt.gca())

X = np.vstack([dataset.X, dataset.Y]).T
Y = dataset.labels_sym.astype(int)
params = opt_param

dev = qml.device("default.qubit", wires=5, analytic=True)#shots=10000, analytic=False)#, shots=100)

# switch to amazon here:
#bucket = "amazon-braket-ionq" # the name of the bucket
#prefix = "example_running_quantum_circuits_on_qpu_devices" # the name of the folder in the bucket
#s3_folder = (bucket, prefix)
#dev_arn = "arn:aws:braket:::device/qpu/rigetti/Aspen-9"
# final safeguard: remove the comment
#dev = qml.device("braket.aws.qubit", device_arn=dev_arn, s3_destination_folder=s3_folder, wires=5, shots=shots, parallel=True)

def layer(x, params, wires, i0=0, inc=1):
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])
        
    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])


@qml.qnode(dev)
def circuit(x, layer_params):
    for j, layer_params in enumerate(params):
        layer(x, layer_params, [0,1,2,3,4], i0=j * 5)
    #layer(x, params[0], [0,1,2,3,4], i0=0*5)
    return [qml.expval(qml.PauliZ(i)) for i in range(5)]

result = circuit(X[0] @ W, params)
print(dev.shots,'n_shots')
print(dev.num_executions,'num executions')
#print(circuit.draw())
print(result)

# +
dataset = DoubleCake(0, 6)
dataset.plot(plt.gca())

X = np.vstack([dataset.X, dataset.Y]).T
Y = dataset.labels_sym.astype(int)
params = opt_param

dev = qml.device("default.qubit", wires=5, analytic=True)#, shots=100)

# switch to amazon here:
#bucket = "amazon-braket-ionq" # the name of the bucket
#prefix = "example_running_quantum_circuits_on_qpu_devices" # the name of the folder in the bucket
#s3_folder = (bucket, prefix)
#dev_arn = "arn:aws:braket:::device/qpu/rigetti/Aspen-9"
# final safeguard: remove the comment
#dev = qml.device("braket.aws.qubit", device_arn=dev_arn, s3_destination_folder=s3_folder, wires=5, shots=shots, parallel=True)


def layer(x, params, wires, i0=0, inc=1):
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])
   
    print(params)
    
    qml.CNOT(wires= wires[:2])
    qml.RZ(-params[1][0]/2, wires=1)
    qml.CNOT(wires= wires[:2])
    qml.RZ(params[1][0]/2, wires=1)
    
    qml.CNOT(wires=wires[1:3])
    qml.RZ(-params[1][1]/2, wires=2)
    qml.CNOT(wires= wires[1:3])
    qml.RZ(params[1][1]/2, wires=2)
        
    qml.CNOT(wires=wires[2:4])
    qml.RZ(-params[1][2]/2, wires=3)
    qml.CNOT(wires= wires[2:4])
    qml.RZ(params[1][2]/2, wires=3)
        
    qml.CNOT(wires=[wires[3],wires[4]])
    qml.RZ(-params[1][3]/2, wires=4)
    qml.CNOT(wires= [wires[3],wires[4]])
    qml.RZ(params[1][3]/2, wires=4)
    
    qml.CNOT(wires=[wires[4],wires[0]])
    qml.RZ(-params[1][4]/2, wires=0)
    qml.CNOT(wires= [wires[4],wires[0]])
    qml.RZ(params[1][4]/2, wires=0)
        
#    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])


@qml.qnode(dev)
def circuit(x, layer_params):
    for j, layer_params in enumerate(params):
        layer(x, layer_params, [0,1,2,3,4], i0=j * 5)
    #layer(x, params[0], [0,1,2,3,4], i0=0*5)
    return [qml.expval(qml.PauliZ(i)) for i in range(5)]


result = circuit(X[0] @ W, params)
print(dev.shots,'n_shots')
print(dev.num_executions,'num executions')
#print(circuit.draw())
print(result)
print(dev.shots)

# +
dataset = DoubleCake(0, 6)
dataset.plot(plt.gca())

X = np.vstack([dataset.X, dataset.Y]).T
Y = dataset.labels_sym.astype(int)
params = opt_param

amazon = False
    
if amazon == True:
    # switch to amazon here:
    bucket = "amazon-braket-ionq" # the name of the bucket
    prefix = "single_evaluation_test" # the name of the folder in the bucket
    s3_folder = (bucket, prefix)
    dev_arn = "arn:aws:braket:::device/qpu/ionq/ionQdevice"
    # final safeguard: remove the comment
    dev = qml.device("braket.aws.qubit", device_arn=dev_arn, s3_destination_folder=s3_folder, wires=5, shots=100, parallel=True)

else: 
    dev = qml.device("default.qubit", wires=5, shots=100, analytic=False)#, shots=100)
    
def layer(x, params, wires, i0=0, inc=1):
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])
        
    qml.CNOT(wires= wires[:2])
    qml.RZ(-params[1][0]/2, wires=1)
    qml.CNOT(wires= wires[:2])
    qml.RZ(params[1][0]/2, wires=1)
    
    qml.CNOT(wires=wires[1:3])
    qml.RZ(-params[1][1]/2, wires=2)
    qml.CNOT(wires= wires[1:3])
    qml.RZ(params[1][1]/2, wires=2)
        
    qml.CNOT(wires=wires[2:4])
    qml.RZ(-params[1][2]/2, wires=3)
    qml.CNOT(wires= wires[2:4])
    qml.RZ(params[1][2]/2, wires=3)
        
    qml.CNOT(wires=[wires[3],wires[4]])
    qml.RZ(-params[1][3]/2, wires=4)
    qml.CNOT(wires= [wires[3],wires[4]])
    qml.RZ(params[1][3]/2, wires=4)
    
    qml.CNOT(wires=[wires[4],wires[0]])
    qml.RZ(-params[1][4]/2, wires=0)
    qml.CNOT(wires= [wires[4],wires[0]])
    qml.RZ(params[1][4]/2, wires=0)


@qml.qnode(dev)
def circuit(x, layer_params):
    for j, layer_params in enumerate(params):
        layer(x, layer_params, [0,1,2,3,4], i0=j * 5)
    return [qml.expval(qml.PauliZ(i)) for i in range(5)]

result = circuit(X[0] @ W, params)
print(dev.shots,'n_shots')
print(dev.num_executions)
#print(circuit.draw())
print(result)
# -

Amazon results:
100 n_shots
0
[ 0.3   0.04 -0.08  0.14 -0.08]

# +
with cnot [ 0.22686291 -0.08029147 -0.0533307  -0.17031097 -0.04753056]
with crz [ 0.22686291 -0.08029147 -0.0533307  -0.17031097 -0.04753056]


