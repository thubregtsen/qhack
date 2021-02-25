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
import tk_lib as tk
from cake_dataset import Dataset as Cake
from cake_dataset import DoubleCake
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm

# +
dataset = DoubleCake(0, 6)
dataset.plot(plt.gca())

X = np.vstack([dataset.X, dataset.Y]).T
Y = dataset.labels_sym.astype(int)


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

bucket = "amazon-braket-5268bd361bba" # the name of the bucket
prefix = "example_running_quantum_circuits_on_qpu_devices" # the name of the folder in the bucket
s3_folder = (bucket, prefix)
#remote_device = qml.device("braket.aws.qubit",device_arn="arn:aws:braket:::device/qpu/rigetti/Aspen-9")

sim = True
if sim:
    print("You're safe")
    dev_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
else:
    print("MONEY IS BEING USED")
    #dev_arn = "arn:aws:braket:::device/qpu/rigetti/Aspen-9"

#dev = qml.device("braket.aws.qubit", device_arn=dev_arn, s3_destination_folder=s3_folder, wires=5, shots=1000, parallel=True)
dev = qml.device("default.qubit",wires=5, shots=100)

#dev = qml.device("default.qubit", wires=5)
wires = list(range(5))
#W = np.random.normal(0, .7, (2, 30))
W = np.array( [[-0.74119257, 0.30086841,-0.14320453,-1.87146484,-1.19334328, 0.20887765,
   0.42185476, 1.07008336, 0.05530981,-1.97887274, 0.05114054, 0.60804905,
  -0.44944269,-0.95144054, 0.72997888, 0.98887196,-0.44504737, 0.89873993,
  -0.04748307,-1.99418412, 0.50444373, 0.72079435, 0.57936803,-0.19192516,
  -0.14326398, 0.3900316 , 0.00493263,-0.65130594,-0.07396585,-0.55452377],
 [-0.15270026, 0.8438305 ,-0.83007426, 0.2557212 , 0.74163553, 0.05994059,
  -0.53729139, 0.70810157, 0.94709195, 0.88405545, 0.36076438,-0.9819268 ,
  -0.84747203,-0.17167618, 0.34466588,-0.73933043,-0.55252627,-0.28033645,
  -0.29080041,-0.61093963,-0.6250839 ,-1.3669767 ,-0.1822461 ,-0.54086508,
  -0.2979392 , 0.7828466 ,-0.40171706, 0.18072282,-0.96469331,-0.11403144]] )
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x @ W, params, wires), dev)

import time

start = time.time()

import tk_lib
params = np.random.uniform(0, 2*np.pi, (6, 2, 5))

svm1 = tk.train_svm(k, X, Y, params)

tk_lib.validate(svm1, X, Y)





#params = init_params
params = np.array( [[[ 2.10033245, 5.55997015, 6.09638606,-0.37144048, 0.63830458],
  [ 0.89884427, 1.81360518, 4.52812214, 3.1859543 ,-0.24917753]], [[ 3.05588035, 1.8706908 , 1.30766142, 4.11245877, 5.36798197],
  [ 0.12250709, 0.90795366, 3.56972424, 1.06983204, 1.82090398]], [[ 3.35208483, 4.63916928, 5.34646727, 2.53980254, 5.39467891],
  [ 4.86007185, 3.9561259 , 1.09033459, 0.68495479,-0.40987279]], [[ 5.36297672, 4.59565098, 1.20311862, 4.04794341, 6.26836612],
  [ 1.8266247 , 3.15363376, 4.04657481, 1.53831804, 2.4819442 ]], [[ 0.52716374, 3.04848144, 5.2174517 , 5.0988528 , 5.58971554],
  [ 3.37993676, 1.19476682, 2.61070494, 0.28040412, 1.57737699]], [[ 2.26992207, 4.42927381, 2.0306127 , 1.48201344, 4.2628975 ],
  [ 5.40116067, 4.7574125 , 5.93709807, 5.990459  , 5.43874762]]] )



svm2 = tk.train_svm(k, X, Y, params)



tk_lib.validate(svm2, X, Y)

end = time.time()

print(end-start)

print("we burned through", dev.num_executions)






