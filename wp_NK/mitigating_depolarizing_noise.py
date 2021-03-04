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

import os 
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from wp_TK import tk_lib as tk


from wp_TK.cake_dataset import Dataset as Cake
from wp_TK.cake_dataset import DoubleCake
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
from sklearn.svm import SVC
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from pennylane_cirq import ops as cirq_ops


# +
dataset = DoubleCake(0, 6)
dataset.plot(plt.gca())

X = np.vstack([dataset.X, dataset.Y]).T
Y = dataset.labels_sym.astype(int)
# -

# # Pretrained optimal parameters

# ## random feature extraction matrix 

W = np.tensor([[-0.48928263,  1.22455895,  0.24524283, -0.90407688, -0.45746766,
          0.36301938, -0.46825804,  1.11975107,  0.32406789, -0.25900904,
          0.52357202,  1.01409992,  0.39667541,  0.03402402,  0.56406879,
         -0.04953572, -1.02760894,  0.74835533,  0.77358223,  0.07167918,
         -0.39601274, -1.20093125, -0.62393424,  0.05618878, -0.79196356,
          0.96112044,  0.01387542,  1.57957725,  0.16040838,  0.51613016],
        [ 0.3960217 ,  0.62389183, -0.61651894, -0.22409924,  0.5818978 ,
          0.6581104 , -0.1050928 , -0.13931877, -0.07393247, -0.57084468,
          0.41364557,  0.21091894, -0.57086992, -0.53807368, -0.87942271,
          0.14083521,  0.57690753,  0.57662288,  1.11345077, -0.86806174,
          0.5386058 , -0.3054007 , -0.20143108,  1.0278662 , -0.041591  ,
         -1.94328892,  1.02577419,  1.06179425,  0.94690698, -0.81254189]], requires_grad=True)

# # optimal parameters

opt_param = np.tensor([[[3.16112692, 2.96383445, 6.42069708, 6.71137123, 2.55598801],
         [2.72606667, 2.99057035, 0.930822  , 2.27364172, 1.55443215]],

        [[3.60626686, 5.6386098 , 2.61898825, 0.0511038 , 2.0884846 ],
         [5.12823881, 2.22767521, 2.38026797, 2.82783246, 3.99380242]],

        [[3.89070753, 1.71989212, 6.32027752, 0.73552391, 2.36183652],
         [1.54754968, 1.07048025, 0.42267783, 4.24899979, 5.05318246]],

        [[2.48488179, 3.26446537, 5.57403376, 2.2393725 , 4.7397544 ],
         [3.51567039, 2.81698389, 6.86245787, 0.5135373 , 3.37328717]],

        [[4.69143899, 1.51311219, 2.04891693, 2.45526122, 5.03910988],
         [4.61716515, 3.81501437, 6.08694709, 2.40819571, 2.90937169]],

        [[4.7955417 , 1.71132909, 3.45214701, 1.30618948, 2.43551656],
         [5.99802411, 0.86416771, 1.52129757, 4.48878166, 5.1649024 ]]], requires_grad=True)


# # Rigetti Circuit

# +
def rigetti_layer(x, params, wires, i0=0, inc=1):
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])
        
    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])


@qml.template
def rigetti_ansatz(x, params, wires):
    for j, layer_params in enumerate(params):
        rigetti_layer(x, layer_params, wires, i0=j * len(wires))
    


# -

# # IonQ Circuit

# +
def ionq_layer(x, params, wires, i0=0, inc=1):
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

@qml.template
def ionq_ansatz(x, params, wires, noise_probability):
    for j, layer_params in enumerate(params):
        ionq_layer(x, layer_params, wires, i0=j * len(wires))
    #for wire in wires:
     #   cirq_ops.Depolarize(noise_probability, wires=wire)
        #qml.AmplitudeDamping(noise_probability, wires=wire)


# -

# # Data from performing this circuit on quantum hardware

kernel_matrix_from_rigetti = np.array([[1., 0.03, 0.02, 0., 0.01, 0.02, 0.01, 0., 0.02, 0.01, 0.02, 0.],
          [0.03, 1., 0.01, 0.01, 0.02, 0.01, 0., 0., 0.01, 0., 0.01, 0.01],
          [0.02, 0.01, 1., 0., 0., 0.03, 0.01, 0.03, 0., 0.02, 0.02, 0.01],
          [0., 0.01, 0., 1., 0.01, 0.01, 0., 0., 0.02, 0.01, 0.03, 0.02],
          [0.01, 0.02, 0., 0.01, 1., 0.01, 0.01, 0., 0., 0.02, 0.04, 0.01],
          [0.02, 0.01, 0.03, 0.01, 0.01, 1., 0., 0., 0.01, 0.01, 0.03, 0.],
          [0.01, 0., 0.01, 0., 0.01, 0., 1., 0.01, 0., 0.01, 0., 0.01],
          [0., 0., 0.03, 0., 0., 0., 0.01, 1., 0.01, 0., 0., 0.03],
          [0.02, 0.01, 0., 0.02, 0., 0.01, 0., 0.01, 1., 0.03, 0., 0.],
          [0.01, 0., 0.02, 0.01, 0.02, 0.01, 0.01, 0., 0.03, 1., 0.01, 0.02],
          [0.02, 0.01, 0.02, 0.03, 0.04, 0.03, 0., 0., 0., 0.01, 1., 0.],
          [0., 0.01, 0.01, 0.02, 0.01, 0., 0.01, 0.03, 0., 0.02, 0., 1.]])

kernel_matrix_from_ionq = np.array([[1., 0.02, 0.01, 0.04, 0.02, 0.06, 0.01, 0., 0.04, 0.02, 0., 0.03],
          [0.02, 1., 0.02, 0.08, 0., 0.03, 0.03, 0.02, 0.02, 0.08, 0.02, 0.04],
          [0.01, 0.02, 1., 0., 0.03, 0.02, 0.06, 0.05, 0.04, 0.04, 0.04, 0.03],
          [0.04, 0.08, 0., 1., 0.02, 0.06, 0.03, 0.06, 0.05, 0.03, 0.05, 0.02],
          [0.02, 0., 0.03, 0.02, 1., 0.03, 0.06, 0.08, 0.01, 0.05, 0.09, 0.03],
          [0.06, 0.03, 0.02, 0.06, 0.03, 1., 0.01, 0.07, 0.04, 0.05, 0.01, 0.08],
          [0.01, 0.03, 0.06, 0.03, 0.06, 0.01, 1., 0.05, 0.06, 0.03, 0.05, 0.04],
          [0., 0.02, 0.05, 0.06, 0.08, 0.07, 0.05, 1., 0.05, 0.04, 0.09, 0.06],
          [0.04, 0.02, 0.04, 0.05, 0.01, 0.04, 0.06, 0.05, 1., 0.05, 0.02, 0.03],
          [0.02, 0.08, 0.04, 0.03, 0.05, 0.05, 0.03, 0.04, 0.05, 1., 0.02, 0.06],
          [0., 0.02, 0.04, 0.05, 0.09, 0.01, 0.05, 0.09, 0.02, 0.02, 1., 0.03],
          [0.03, 0.04, 0.03, 0.02, 0.03, 0.08, 0.04, 0.06, 0.03, 0.06, 0.03, 1.]])

kernel_matrix_from_simulation = np.array([[1., 0.03, 0.05, 0., 0.23, 0.1, 0.26, 0.39, 0., 0.3, 0.06, 0.15],
          [0.03, 1., 0.07, 0.22, 0.03, 0.13, 0.18, 0., 0.29, 0.02, 0.38, 0.02],
          [0.05, 0.07, 1., 0.05, 0.07, 0.05, 0.23, 0.42, 0.53, 0.07, 0.19, 0.13],
          [0., 0.22, 0.05, 1., 0., 0.07, 0.26, 0.05, 0.11, 0.28, 0.38, 0.03],
          [0.23, 0.03, 0.07, 0., 1., 0.04, 0.02, 0.29, 0.03, 0.17, 0.01, 0.15],
          [0.1, 0.13, 0.05, 0.07, 0.04, 1., 0.07, 0.14, 0.14, 0.07, 0.34, 0.45],
          [0.26, 0.18, 0.23, 0.26, 0.02, 0.07, 1., 0.47, 0.36, 0.15, 0.25, 0.],
          [0.39, 0., 0.42, 0.05, 0.29, 0.14, 0.47, 1., 0.23, 0.3, 0.09, 0.23],
          [0., 0.29, 0.53, 0.11, 0.03, 0.14, 0.36, 0.23, 1., 0.01, 0.23, 0.02],
          [0.3, 0.02, 0.07, 0.28, 0.17, 0.07, 0.15, 0.3, 0.01, 1., 0.49, 0.39],
          [0.06, 0.38, 0.19, 0.38, 0.01, 0.34, 0.25, 0.09, 0.23, 0.49, 1., 0.31],
          [0.15, 0.02, 0.13, 0.03, 0.15, 0.45, 0., 0.23, 0.02, 0.39, 0.31, 1.]])


# # Simulating the circuit with noise 

# +
class noisy_kernel(qml.kernels.EmbeddingKernel):
    """add noise to circuit"""
    def __init__(self,ansatz, device, noise_probability, interface="autograd", diff_method="best", **kwargs):
        self.probs_qnode = None
        """QNode: The QNode representing the quantum embedding kernel."""
        self.noise_probability = noise_probability
        print(self.noise_probability)

        def circuit(x1, x2, params,**kwargs):
            ansatz(x1, params, **kwargs)
            qml.inv(ansatz(x2, params, **kwargs))
            for wire in device.wires:
                
                cirq_ops.Depolarize(self.noise_probability, wires=wire)
            return qml.probs(wires=device.wires)
        
        self.probs_qnode = qml.QNode(
            circuit, device, interface=interface, diff_method=diff_method, **kwargs
        )
        

# -

def calculate_kernel_matrix(noise_probability, hardware_backend):
    shots = 100
    local = False
    dev = qml.device("cirq.mixedsimulator", wires=5, shots=shots, analytic=False)

    df = pd.DataFrame()
    wires = list(range(5))
    k = noisy_kernel(lambda x, params: rigetti_ansatz(x @ W, params, wires), dev, noise_probability )

    name_params = 'optimal'
    init_params = opt_param
    
    K_raw1 = k.square_kernel_matrix(X, init_params) # Use quantum computer
    return(K_raw1)


def visualize_kernel_matrices(kernel_matrices, noise_probabilites):
    fig,ax = plt.subplots(1, len(kernel_matrices))
    print(fig,'ax')
    sns.set()
    for i, kernel_matrix in enumerate(kernel_matrices):
        sns.heatmap(kernel_matrix, vmin=0, vmax=1, xticklabels='', yticklabels='', ax=ax[i], cmap='Spectral', cbar=False)
    


noise_probabilities_array = (np.arange(0, 1, 0.2))
kernel_matrices = []
for noise_p in noise_probabilities_array:
    print(noise_p)
    k_matrix = calculate_kernel_matrix(noise_p, 'rigetti')
    kernel_matrices.append(k_matrix)


print(noise_probabilities_array)
visualize_kernel_matrices(kernel_matrices, noise_probabilities_array)


# +
def my_quantum_function(x, y):
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0,1])
    qml.RY(y, wires=1)
    #cirq_ops.Depolarize(0.1, wires=1)
    return qml.expval(qml.PauliZ(1))
dev = qml.device('cirq.mixedsimulator', wires=2, shots=1000, analytic=True)

circuit = qml.QNode(my_quantum_function, dev)
circuit(1,2)
# -


