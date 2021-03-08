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
# -


# # Some global variables (used below!)

shots = 0
#  Actually use shots by setting analytic to False in the device. If shots=0, use analytic
analytic_device = (shots==0)
num_wires = 5

# # Dataset

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
def apply_noise(ansatz, device, noise_channel, args_noise_channel):
    
    def noisy_ansatz(*args_ansatz, **kwargs_ansatz):
    
        def _ansatz(*args, **kwargs):
            ansatz(*args, **kwargs)
            return qml.expval(qml.PauliZ(0))

        _qnode = qml.QNode(_ansatz, device)
        _qnode.construct(args_ansatz, kwargs_ansatz)
        ops = _qnode.qtape.operations
    
        for op in ops:
            op.__class__(*op.data, wires=op.wires)
            noise_channel(*args_noise_channel, wires=op.wires)
    
    return noisy_ansatz


class noisy_kernel(qml.kernels.EmbeddingKernel):
    """adds noise to a QEK
    Args:
      ansatz (callable): See qml.kernels.EmbeddingKernel
      device (qml.Device, Sequence[Device]): See qml.kernels.EmbeddingKernel
      noise_channel (qml.Operation): Noise channel to be applied on the level of noise_application_level.
      args_noise_channel (dict): arguments to be passed to the noise_channel.
      noise_application_level (str):
        'global': Apply noise after the full circuit.
        'per_embedding': Apply noise after the embedding and the inverted embedding each.
        'per_gate': Apply noise after each gate (incompatible with noise_channels acting on all qubits).
    """
    
    def __init__(self, ansatz, device, noise_channel, args_noise_channel, noise_application_level, **kwargs):
        self.probs_qnode = None
        self.noise_channel = noise_channel
        self.args_noise_channel = args_noise_channel
        self.noise_application_level = noise_application_level
        if self.noise_application_level == 'per_gate':
            self.noisy_ansatz = apply_noise(ansatz, device, noise_channel, args_noise_channel)
        else:
            self.noisy_ansatz = ansatz
        

        def circuit(x1, x2, params,**kwargs):
            self.noisy_ansatz(x1, params, **kwargs)
            if self.noise_application_level == 'per_embedding':
                self.noise_channel(*self.args_noise_channel)
                
            self.noisy_ansatz(x2, params, **kwargs)
            if self.noise_application_level in ('per_embedding', 'global'):
                self.noise_channel(*self.args_noise_channel)

            return qml.probs(wires=device.wires)
        
        self.probs_qnode = qml.QNode(circuit, device, **kwargs)
        
class singly_qubit_noisy_kernel(qml.kernels.EmbeddingKernel):
    """adds single-qubit noise to the end of the circuit for ancilla-free approach."""
    def __init__(self,ansatz, device, noise_probability, **kwargs):
        self.probs_qnode = None
        self.noise_probability = noise_probability
        print(self.noise_probability)

        def circuit(x1, x2, params,**kwargs):
            ansatz(x1, params, **kwargs)
            qml.inv(ansatz(x2, params, **kwargs))
            for wire in device.wires:
                cirq_ops.Depolarize(self.noise_probability, wires=wire)
            return qml.probs(wires=device.wires)
        
        self.probs_qnode = qml.QNode(circuit, device, **kwargs)
        
class depolarize_global_kernel(qml.kernels.EmbeddingKernel):
    """effectively adds global noise after the entire circuit, for testing purposes"""
    def __init__(self,ansatz, device, lambda_, **kwargs):
        self.probs_qnode = None
        self.lambda_ = lambda_
        
        def circuit(x1, x2, params,**kwargs):
            ansatz(x1, params, **kwargs)
            qml.inv(ansatz(x2, params, **kwargs))
            return qml.probs(wires=device.wires)
        
        def wrapped_qnode(*args, **qnode_kwargs):
            qnode_ = qml.QNode(circuit, device, **kwargs)(*args, **qnode_kwargs)
            return (1-self.lambda_) * qnode_ +self.lambda_/(2**device.num_wires)       
        
        self.probs_qnode = wrapped_qnode
        
class depolarize_per_embedding_kernel(qml.kernels.EmbeddingKernel):
    """effectively adds global noise after each embedding with individual noise rates, for testing purposes"""
    def __init__(self,ansatz, device, lambdas, X_, **kwargs):
        """This kernel requires the training set as input in order to properly emulate the noise model"""
        self.probs_qnode = None
        self.lambdas = np.array(lambdas)
        self.X_ = X_
        
        def circuit(x1, x2, params,**kwargs):
            ansatz(x1, params, **kwargs)
            qml.inv(ansatz(x2, params, **kwargs))
            return qml.probs(wires=device.wires)
        
        def wrapped_qnode(x1, x2, *qnode_args, **qnode_kwargs):
            qnode_ = qml.QNode(circuit, device, **kwargs)(x1, x2, *qnode_args, **qnode_kwargs)
            lambda_1 = self.lambdas[np.where([np.allclose(x_, x1) for x_ in self.X_])[0]]
            lambda_2 = self.lambdas[np.where([np.allclose(x_, x2) for x_ in self.X_])[0]]
            lambda_ = - (1-lambda_1) * (1-lambda_2) + 1
            return (1-lambda_) * qnode_ +lambda_/(2**device.num_wires)       
        
        self.probs_qnode = wrapped_qnode


# +
# def calculate_kernel_matrix(noise_channel, fix_diag=True):
#     local = False
#     if shots==0:
#         shots_device = 1 # shots=0 raises an error...
#     dev = qml.device("cirq.mixedsimulator", wires=num_wires, shots=shots_device, analytic=analytic_device)

#     wires = list(range(5))
#     # Here one can choose the noisy kernel model
#     k = noisy_kernel(lambda x, params: rigetti_ansatz(x @ W, params, wires), dev, noise_probability )
# #     k = dummy_kernel(lambda x, params: rigetti_ansatz(x @ W, params, wires), dev, lambda_=noise_probability )
# #     lambdas = (0.8+np.random.random(len(X))*0.2)*noise_probability
# #     k = dummy_split_kernel(lambda x, params: rigetti_ansatz(x @ W, params, wires), dev, lambdas=lambdas, X_=X)

# #     name_params = 'optimal'
#     init_params = opt_param
#     k_mapped = lambda x1, x2: k(x1, x2, init_params)
#     K_raw1 = qml.kernels.square_kernel_matrix(X, k_mapped, assume_normalized_kernel=fix_diag)
# #     K_raw1 = k.square_kernel_matrix(X, init_params) # Use quantum computer
#     return K_raw1
# -

def visualize_kernel_matrices(kernel_matrices, noise_probabilites, draw_last_cbar=False):
    num_mat = len(kernel_matrices)
    width_ratios = [1]*num_mat+[0.2]*int(draw_last_cbar)
    fig,ax = plt.subplots(1, num_mat+draw_last_cbar, figsize=(num_mat*5+draw_last_cbar, 5), gridspec_kw={'width_ratios': width_ratios})
    sns.set()
    for i, kernel_matrix in enumerate(kernel_matrices):
        plot = sns.heatmap(
            kernel_matrix, 
            vmin=0,
            vmax=1,
            xticklabels='',
            yticklabels='',
            ax=ax[i],
            cmap='Spectral',
            cbar=False
        )
    if draw_last_cbar:
        ch = plot.get_children()
        fig.colorbar(ch[0], ax=ax[-2], cax=ax[-1])


def mitigate_global_depolarization(kernel_matrix, num_wires, strategy='average', use_entries=None):
    """Estimate the noise rate of a global depolarizing noise model based on the diagonal entries of a kernel
    matrix and mitigate the effect of said noise model.
    Args:
      kernel_matrix (ndarray): Noisy kernel matrix.
      num_wires (int): Number of wires/qubits that was used to compute the kernel matrix.
      strategy ('average'|'split_channel'): Details of the noise model and strategy for mitigation.
        'average': Compute the noise rate based on the diagonal entries in use_entries, average if applicable.
        'split_channel': Assume a distinct effective noise rate for the embedding circuit of each feature vector.
      use_entries (list<int>): Indices of diagonal entries to use if strategy=='average'. Set to all if None.
    Returns:
      mitigated_matrix (ndarray): Mitigated kernel matrix.
      noise_rates (ndarray): Determined noise rates, meaning depends on kwarg strategy.
    Comments:
      If strategy is 'average', the diagonal entries with indices use_entries have to be measured on the QC.
      If it is 'split_channel', all diagonal entries are required.
    """
    dim = 2**num_wires
    
    if strategy=='average':
        if use_entries is None:
            diagonal_elements = np.diag(kernel_matrix)
        else:
            diagonal_elements = np.diag(kernel_matrix)[use_entries]
        noise_rates = (1 - diagonal_elements) * dim / (dim - 1)
        eff_noise_rate = np.mean(noise_rates)
        mitigated_matrix = (kernel_matrix - eff_noise_rate / dim) / (1 - eff_noise_rate)
        
    elif strategy=='split_channel':
        noise_rates = (1 - np.diag(kernel_matrix)) * dim / (dim - 1)
        noise_rates = 1-np.sqrt(1-noise_rates)
        n = len(kernel_matrix)
        inverse_noise = -np.outer(noise_rates, noise_rates)\
            + noise_rates.reshape((1, n))\
            + noise_rates.reshape((n, 1))
        mitigated_matrix = (kernel_matrix - inverse_noise / dim) / (1 - inverse_noise)
    
    return mitigated_matrix, noise_rates


# # Noise mitigation computations
#
# ## Compute noisy kernel matrix

# +
noise_probabilities = np.arange(0, 0.8, 0.2)
kernel_matrices = []
fix_diag = False # Compute the diagonal entries
if shots==0:
    shots_device = 1 if shots==0 else shots # shots=0 raises an error...

dev = qml.device("cirq.mixedsimulator", wires=num_wires, shots=shots_device, analytic=analytic_device)
rigetti_ansatz_mapped = lambda x, params: rigetti_ansatz(x @ W, params, range(num_wires))
noise_channel = lambda p, wires: [cirq_ops.Depolarize(p, wires=wire) for wire in wires]
    
for noise_p in noise_probabilities:
    print(noise_p)
    k = noisy_kernel(
        rigetti_ansatz_mapped,
        dev,
        noise_channel=noise_channel,
        args_noise_channel=(noise_p,),
        noise_application_level='per_gate',
    )
    k_mapped = lambda x1, x2: k(x1, x2, opt_param)
    
    K_raw1 = qml.kernels.square_kernel_matrix(X, k_mapped, assume_normalized_kernel=fix_diag)    
        
    kernel_matrices.append(k_matrix)
# -


# ## Compute noise-mitigated matrices
# We look at three strategies of mitigation, corresponding to distinct assumptions on the noise.

mitigated_matrices = {
    (strategy, use_entries): 
    [
    mitigate_global_depolarization(K, num_wires=num_wires, strategy=strategy, use_entries=use_entries)[0] 
        for K in kernel_matrices
]
    for strategy, use_entries in [('average', (0,)), ('average', None), ('split_channel', None)]
}

# ## Plotting mitigated noise
# In the following collection of heatmaps, the row corresponds to the mitigation technique:
#  0. No mitigation
#  1. Based on first diagonal element
#  2. Averaged over entire diagonal
#  3. split up noise per embedding (and inverse embedding) subcircuits
#
#  The columns correspond to the noise strength
#  0. No noise
#  1. noise_probability=0.2
#  2. noise_probability=0.4
#  3. noise_probability=0.6
#  4. noise_probability=0.8
#
#  The used noisy kernel (there are two for testing, namely `dummy_kernel` and `dummy_split_kernel`) is set in `calculate_kernel_matrix`.
#  
#  Note that for testing, `analytic` has to be set to `False` in the device because otherwise statistical 
#  fluctuations will prevent a perfect mitigation. 

print(noise_probabilities)
visualize_kernel_matrices(kernel_matrices, noise_probabilities, draw_last_cbar=True)
for mats in mitigated_matrices.values():
    visualize_kernel_matrices(mats, noise_probabilities, draw_last_cbar=True)

# +
np.set_printoptions(precision=5)
distances = np.zeros((len(mitigated_matrices)+1, len(noise_probabilities)))
violation = np.zeros((len(mitigated_matrices)+1, len(noise_probabilities)))

for j, mat in enumerate(kernel_matrices):
    distances[0,j] = np.linalg.norm(mat-kernel_matrices[0], 'fro')
    violation[0,j] = np.linalg.eigvalsh(mat)[0]
#     distances[i+1,j] = np.max(np.abs(mat-kernel_matrices[0]))
for i, (key, mats) in enumerate(mitigated_matrices.items()):
    for j, mat in enumerate(mats):
        distances[i+1,j] = np.linalg.norm(mat-kernel_matrices[0], 'fro')
        violation[i+1,j] = np.linalg.eigvalsh(mat)[0]
#         distances[i+1,j] = np.max(np.abs(mat-kernel_matrices[0]))
print(distances)
print(violation)

# +
# Deactivated this cell for now :-)

# def my_quantum_function(x, y):
#     qml.RZ(x, wires=0)
#     qml.CNOT(wires=[0,1])
#     qml.RY(y, wires=1)
#     #cirq_ops.Depolarize(0.1, wires=1)
#     return qml.expval(qml.PauliZ(1))
# dev = qml.device('cirq.mixedsimulator', wires=2, shots=1000, analytic=True)

# circuit = qml.QNode(my_quantum_function, dev)
# circuit(1,2)
# +
dev = qml.device("cirq.mixedsimulator", wires=num_wires, shots=100, analytic=analytic_device)

def ans(*args, **kwargs):
    rigetti_ansatz(*args, **kwargs)
    cirq_ops.Depolarize(0.1, wires=range(num_wires))
    return qml.expval(qml.PauliZ(0))
    
q = qml.QNode(
    ans,
    dev)

# -


print(dev.state)
q.construct((X[0], opt_param), {'wires': range(num_wires)})
# q(X[0], opt_param, wires=range(num_wires))
print(dev.state)

op = q.qtape.operations[4]

op.__class__(*op.data, wires=op.wires)




