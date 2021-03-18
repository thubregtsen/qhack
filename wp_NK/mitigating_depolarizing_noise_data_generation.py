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
import numpy as pure_np
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
# from sklearn.svm import SVC
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from pennylane_cirq import ops as cirq_ops
import nk_lib
from itertools import product
from dill import dump
# -


# # Some global variables (used below!)

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
    N = len(wires)
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
    for j, wire in enumerate(wires):    
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
    for j, wire in enumerate(wires):
        qml.RY(params[0, j], wires=[wire])
        
    for j in range(N//2):
        qml.CRZ(params[1][2*j], wires=wires[2*j:2*(j+1)])
    for j in range(N//2):
        qml.CRZ(params[1][2*j+1], wires=[wires[2*j+1], wires[(2*j+2)%N]])
    if N%2==1: # There is the "hop the boundary" gate missing if N is odd.
        qml.CRZ(params[1][-1], wires=[wires[-1], wires[0]])


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

# # Simulate noise

# ## Circuit Level Noise single qubit depolarizing channel

# +
# def apply_circuit_level_noise(ansatz, device, noise_channel, args_noise_channel, adjoint=False):
#     """Add noise channels after each gate of an ansatz
#     Args:
#       ansatz (qml.template): Ansatz to add noise to.
#       device (qml.Device): Device that the ansatz will be run on.
#       noise_channel (qml.Operation): Noise channel to apply after each gate;
#         Note that it should be applicable on a flexible number of qubits.
#       args_noise_channel (tuple): Arguments to pass to noise_channel.
#       adjoint (bool): Whether the adjoint/inverse of the circuits should be used.
#     Returns:
#       noisy_ansatz (qml.template): The ansatz with added noise.
#     """
    
#     # Dummy device to construct the qnode on.
#     # This is an additional safety measure to avoid changing the device state with the QNode.construct call.
#     _device = device.__class__(wires=device.wires)

#     def noisy_ansatz(*args_ansatz, **kwargs_ansatz):
#         # Define a circuit with which a QNode can be instantiated.
#         def _ansatz(*args, **kwargs):
#             if adjoint:
#                 qml.inv(ansatz(*args, **kwargs))
#             else:
#                 ansatz(*args, **kwargs)
#             return qml.expval(qml.PauliZ(0))

#         # Instantiate and construct a qnode and extract the list of operations.
#         _qnode = qml.QNode(_ansatz, _device)
#         _qnode.construct(args_ansatz, kwargs_ansatz)
#         ops = _qnode.qtape.operations        
# #         print(ops)
#         timestep = 0
#         active_qubits = set()
#         for op in ops:
#             qml.QueuingContext.remove(op)
#             if set(op.wires).intersection(active_qubits):
#                 # The operator we are about to apply advances the time step
# #                 angle = op.data[0] if op.data else 3*np.pi/2
#                 inactive_wires = list( set(device.wires).difference(active_qubits))
#                 # Set idling gate noise to half of the maximal strength
#                 noise_channel(*args_noise_channel, np.pi, wires=inactive_wires)
#                 active_qubits = set()
#                 timestep +=1
            
#             if op.inverse:
#                 op.__class__(*op.data, wires=op.wires).inv()
#             else:
#                 op.__class__(*op.data, wires=op.wires)
            
#             # op either has a rotation angle or is a Hadamard, which is RX(np.pi) @ RY(np.pi/2)
#             angle = op.data[0] if op.data else 3*np.pi/2
#             noise_channel(*args_noise_channel, angle, wires=op.wires)
#             active_qubits |= set(op.wires)

#     return noisy_ansatz


# +
# def apply_noise(ansatz, device, noise_channel, args_noise_channel, adjoint=False):
#     """Add noise channels after each gate of an ansatz
#     Args:
#       ansatz (qml.template): Ansatz to add noise to.
#       device (qml.Device): Device that the ansatz will be run on.
#       noise_channel (qml.Operation): Noise channel to apply after each gate;
#         Note that it should be applicable on a flexible number of qubits.
#       args_noise_channel (tuple): Arguments to pass to noise_channel.
#       adjoint (bool): Whether the adjoint/inverse of the circuits should be used.
#     Returns:
#       noisy_ansatz (qml.template): The ansatz with added noise.
#     """
    
#     # Dummy device to construct the qnode on.
#     # This is an additional safety measure to avoid changing the device state with the QNode.construct call.
    
    
#     @qml.template
#     def noisy_ansatz(*args_ansatz, **kwargs_ansatz):
#         _device = device.__class__(wires=device.wires)
#         # Define a circuit with which a QNode can be instantiated.
#         def _ansatz(*args, **kwargs):
#             if adjoint:
#                 qml.inv(ansatz(*args, **kwargs))
#             else:
#                 ansatz(*args, **kwargs)
#             return qml.expval(qml.PauliZ(0))

#         # Instantiate and construct a qnode and extract the list of operations.
#         _qnode = qml.QNode(_ansatz, _device)
#         _qnode.construct(args_ansatz, kwargs_ansatz)
#         ops = _qnode.qtape.operations
        
#         # Apply the original operations and the noise_channel alternatingly.
#         for op in ops:
#             qml.QueuingContext.remove(op)
#             if op.inverse:
#                 op.__class__(*op.data, wires=op.wires).inv()
#             else:
#                 op.__class__(*op.data, wires=op.wires)
#             angle = op.data[0] if op.data else 3*np.pi/2
#             noise_channel(*args_noise_channel, angle, wires=op.wires)
    
#     return noisy_ansatz


        
# class single_qubit_noisy_kernel(qml.kernels.EmbeddingKernel):
#     """adds single-qubit noise to the end of the circuit for ancilla-free approach."""
#     def __init__(self,ansatz, device, noise_probability, **kwargs):
#         self.probs_qnode = None
#         self.noise_probability = noise_probability

#         def circuit(x1, x2, params,**kwargs):
#             ansatz(x1, params, **kwargs)
#             qml.inv(ansatz(x2, params, **kwargs))
#             for wire in device.wires:
#                 cirq_ops.Depolarize(self.noise_probability, wires=wire)
#             return qml.probs(wires=device.wires)
        
#         self.probs_qnode = qml.QNode(circuit, device, **kwargs)
        
# class depolarize_global_kernel(qml.kernels.EmbeddingKernel):
#     """effectively adds global noise after the entire circuit, for testing purposes"""
#     def __init__(self,ansatz, device, lambda_, **kwargs):
#         self.probs_qnode = None
#         self.lambda_ = lambda_
        
#         def circuit(x1, x2, params,**kwargs):
#             ansatz(x1, params, **kwargs)
#             qml.inv(ansatz(x2, params, **kwargs))
#             return qml.probs(wires=device.wires)
        
#         def wrapped_qnode(*args, **qnode_kwargs):
#             qnode_ = qml.QNode(circuit, device, **kwargs)(*args, **qnode_kwargs)
#             return (1-self.lambda_) * qnode_ +self.lambda_/(2**device.num_wires)       
        
#         self.probs_qnode = wrapped_qnode
        
# class depolarize_per_embedding_kernel(qml.kernels.EmbeddingKernel):
#     """effectively adds global noise after each embedding with individual noise rates, for testing purposes"""
#     def __init__(self,ansatz, device, lambdas, X_, **kwargs):
#         """This kernel requires the training set as input in order to properly emulate the noise model"""
#         self.probs_qnode = None
#         self.lambdas = np.array(lambdas)
#         self.X_ = X_
        
#         def circuit(x1, x2, params,**kwargs):
#             ansatz(x1, params, **kwargs)
#             qml.inv(ansatz(x2, params, **kwargs))
#             return qml.probs(wires=device.wires)
        
#         def wrapped_qnode(x1, x2, *qnode_args, **qnode_kwargs):
#             qnode_ = qml.QNode(circuit, device, **kwargs)(x1, x2, *qnode_args, **qnode_kwargs)
#             lambda_1 = self.lambdas[np.where([np.allclose(x_, x1) for x_ in self.X_])[0]]
#             lambda_2 = self.lambdas[np.where([np.allclose(x_, x2) for x_ in self.X_])[0]]
#             lambda_ = - (1-lambda_1) * (1-lambda_2) + 1
#             return (1-lambda_) * qnode_ +lambda_/(2**device.num_wires)       
        
#         self.probs_qnode = wrapped_qnode
# -

def noise_channel(base_p, data=None, wires=None):
    if data is None:
        # Idling gate
        angle = np.pi
    elif len(data)==0:
        # Non-parametrized gate ( Hadamard in this circuit )
        angle = 3 * np.pi / 2
    else:
        # Parametrized gate
        angle = data[0]
        
    relative_p = base_p * ( np.abs(angle) / (2*np.pi) )
    noise_ops = [cirq_ops.Depolarize(relative_p, wires=wire) for wire in wires]
    return noise_ops


# +

# k.probs_qnode.construct((X[0], X[0], opt_param), {})
# k.probs_qnode.qtape.operations
# -

# # Noise mitigation computations
#
# ## Compute noisy kernel matrix

# +
fix_diag = False # Compute the diagonal entries for mitigation.
rigetti_ansatz_mapped = lambda x, params: rigetti_ansatz(x @ W, params, range(num_wires))

shot_numbers = [10, 100, 1000, 10000, 0]
# shot_numbers = [10]
noise_probabilities = np.arange(0.05, 0.1, 0.002)

kernel_matrices = {}
for noise_p, shots in tqdm.notebook.tqdm(product(noise_probabilities, shot_numbers)):
    analytic_device = (shots==0)
    shots_device = 1 if shots==0 else shots # shots=0 raises an error...

    dev = qml.device("cirq.mixedsimulator", wires=num_wires, shots=shots_device, analytic=analytic_device)
    k = nk_lib.noisy_kernel(
        rigetti_ansatz_mapped,
        dev,
        noise_channel=noise_channel,
        args_noise_channel=(noise_p,),
        noise_application_level='per_gate',
    )
    k_mapped = lambda x1, x2: k(x1, x2, opt_param)
    
    K = qml.kernels.square_kernel_matrix(X, k_mapped, assume_normalized_kernel=fix_diag)       
    kernel_matrices[(float(noise_p), shots)] = K

# -


# ## Compute noise-mitigated matrices
# We look at three strategies of mitigation, corresponding to distinct assumptions on the noise.

# +
# mitigated_matrices = {
#     (strategy, use_entries): 
#     [
#     nk_lib.mitigate_global_depolarization(K, num_wires=num_wires, strategy=strategy, use_entries=use_entries)[0] 
#         for K in kernel_matrices
# ]
#     for strategy, use_entries in [(None, None), ('average', (0,)), ('average', None), ('split_channel', None)]
# }

# def wrap_mitigation(mat, fun=qml.kernels.displace_matrix, **kwargs):
#     try:
#         return fun(mat, **kwargs)
#     except:
#         return None

# fun = nk_lib.closest_psd_matrix    
    
# doubly_mitigated_matrices = {
#     (strategy, use_entries): 
#     [
#     wrap_mitigation(
#         nk_lib.mitigate_global_depolarization(K, num_wires=num_wires, strategy=strategy, use_entries=use_entries)[0],
#         fun=fun,
#         fix_diag=True,
#     ) 
#         for K in kernel_matrices
# ]
#     for strategy, use_entries in [(None, None), ('average', (0,)), ('average', None), ('split_channel', None)]
# }
# -

# # Save data

# +
filename = f'data/kernel_matrices_2d_sweep_more_noise.dill'
pure_np_kernel_matrices = {key: pure_np.asarray(mat) for key, mat in kernel_matrices.items()}
dump(pure_np_kernel_matrices, open(filename, 'wb+'))
# np.save(f'kernel_matrices_{file_name_add}.npy', kernel_matrices, allow_pickle = True)

# file_name_add = "analytic" if analytic_device else str(shots_device)
# mitigated_matrices_arrays = {}
# for ki in mitigated_matrices.keys():
#     mitigated_matrices_arrays[ki] = []
#     for valu in mitigated_matrices[ki]:
#         mitigated_matrices_arrays[ki].append(valu if valu is None else valu.numpy())
# print('change save file name!')
# np.save(f'mitigated_matrices_{file_name_add}.npy', mitigated_matrices_arrays,allow_pickle=True)

# doubly_mitigated_matrices_arrays = {}
# for ki in doubly_mitigated_matrices.keys():
#     doubly_mitigated_matrices_arrays[ki] = []
#     for valu in doubly_mitigated_matrices[ki]:
#         doubly_mitigated_matrices_arrays[ki].append(valu if valu is None else valu.numpy())

# np.save(f'doubly_mitigated_matrices_{file_name_add}_{fun.__name__}.npy', doubly_mitigated_matrices_arrays,allow_pickle=True)
# print('change save file name!')

# +
# noise_p = 0.01
# kernel_matrices = []
# fix_diag = False # Compute the diagonal entries
# shots_device = 1 if shots==0 else shots # shots=0 raises an error...
# import pyquil
# p=0.01
# no_error_prob = [1-p]
# error_prob = [p/15]*15
# probabilities = no_error_prob + error_prob
# kraus_operators_depolarizing = pyquil.noise.pauli_kraus_map(probabilities)
# rigetti_ansatz_mapped = lambda x, params: rigetti_ansatz(x @ W, params, range(num_wires))
# #noise_channel = lambda p, wires: [cirq_ops.Depolarize(p, wires=wire) for wire in wires]
# from pennylane.operation import Operation

# from pennylane_cirq import ops


# noise_channel = lambda p, wires: [ops.Depolarize(p, wires=wire) for wire in wires]

# #noise_channel_2_qubits = lambda p, wires: qml.Depolarize(p, wires=wires)
# #noise_channel_2_qubits.num_wires = 2

# dev = qml.device("cirq.mixedsimulator", wires=num_wires, shots=shots_device, analytic=analytic_device)
# k = noisy_kernel(
#     rigetti_ansatz_mapped,
#     dev,
#     noise_channel=noise_channel,
#     args_noise_channel=(noise_p,),
#     noise_application_level='circuit_level',
# )
# #     k = qml.kernels.EmbeddingKernel(rigetti_ansatz_mapped, dev) # Noise-free, for testing
# k_mapped = lambda x1, x2: k(x1, x2, opt_param)

# K_raw1 = qml.kernels.square_kernel_matrix(X, k_mapped, assume_normalized_kernel=fix_diag) 
# -

# # Attempt at circuit level noise with two qubit depolarizing channel

def apply_circuit_level_two_qubit_noise(ansatz, device, noise_channel, args_noise_channel, adjoint=False):
    """Add noise channels after each gate of an ansatz
    Args:
      ansatz (qml.template): Ansatz to add noise to.
      device (qml.Device): Device that the ansatz will be run on.
      noise_channel (qml.Operation): Noise channel to apply after each gate;
        Note that it should be applicable on a flexible number of qubits.
      args_noise_channel (tuple): Arguments to pass to noise_channel.
      adjoint (bool): Whether the adjoint/inverse of the circuits should be used.
    Returns:
      noisy_ansatz (qml.template): The ansatz with added noise.
    """
    
    # Dummy device to construct the qnode on.
    # This is an additional safety measure to avoid changing the device state with the QNode.construct call.
    _device = device.__class__(wires=device.wires)
    print('testing')
    @qml.template
    def noisy_ansatz(*args_ansatz, **kwargs_ansatz):
        
        # Define a circuit with which a QNode can be instantiated.
        def _ansatz(*args, **kwargs):
            #if adjoint:
            #    qml.inv(ansatz(*args, **kwargs))
            
            ansatz(*args, **kwargs)
            return qml.expval(qml.PauliZ(0))

        # Instantiate and construct a qnode and extract the list of operations.
        _qnode = qml.QNode(_ansatz, _device)
        _qnode.construct(args_ansatz, kwargs_ansatz)
        ops = _qnode.qtape.operations
        
        #print(_qnode.qtape.graph.operations_in_order)
        # Apply the original operations and the noise_channel alternatingly.
        # does this add idling noise?
        
        for op in ops:
            #if op.inverse:
              #  op.__class__(*op.data, wires=op.wires).inv()
            #else:
            op.__class__(*op.data, wires=op.wires)

            #print(noise_channel,'noise_channel')
            #if set(op.wires).intersection(active_qubits_in_timestep):
                
            #else:
            #    active_qubits_in_timestep.union(op.wires)
            if len(op.wires) == 2:
                noise_channel[1](*args_noise_channel, wires= op.wires)
            else:
                noise_channel[0](*args_noise_channel, wires= op.wires)
        
        
    return noisy_ansatz


# +
# noise_p = 0.01
# kernel_matrices = []
# fix_diag = False # Compute the diagonal entries
# shots_device = 1 if shots==0 else shots # shots=0 raises an error...
# import pyquil
# p=0.01
# no_error_prob = [1-p]
# error_prob = [p/15]*15
# probabilities = no_error_prob + error_prob
# kraus_operators_depolarizing = pyquil.noise.pauli_kraus_map(probabilities)
# rigetti_ansatz_mapped = lambda x, params: rigetti_ansatz(x @ W, params, range(num_wires))
# #noise_channel = lambda p, wires: [cirq_ops.Depolarize(p, wires=wire) for wire in wires]
# from pennylane.operation import Operation




# noise_channel_2_qubits = lambda p, wires: Depolarize(p, wires=wires)
# noise_channel_2_qubits.num_wires = 2

# dev = qml.device("cirq.mixedsimulator", wires=num_wires, shots=shots_device, analytic=analytic_device)
# k = noisy_kernel(
#     rigetti_ansatz_mapped,
#     dev,
#     noise_channel=[noise_channel, noise_channel_2_qubits],
#     args_noise_channel=(noise_p,),
#     noise_application_level='circuit_level',
# )
# #     k = qml.kernels.EmbeddingKernel(rigetti_ansatz_mapped, dev) # Noise-free, for testing
# k_mapped = lambda x1, x2: k(x1, x2, opt_param)

# K_raw1 = qml.kernels.square_kernel_matrix(X, k_mapped, assume_normalized_kernel=fix_diag) 

# +
# noise_p = 0.01
# kernel_matrices = []
# fix_diag = False # Compute the diagonal entries
# shots_device = 1 if shots==0 else shots # shots=0 raises an error...
# #dev = qml.device("default.qubit", wires=num_wires, shots=shots_device, analytic=analytic_device)
# import pyquil
# p=0.01
# no_error_prob = [1-p]
# error_prob = [p/15]*15
# probabilities = no_error_prob + error_prob
# k_matrices = pyquil.noise.pauli_kraus_map(probabilities)

# rigetti_ansatz_mapped = lambda x, params: rigetti_ansatz(x @ W, params, range(num_wires))
# noise_channel = lambda p, wires: [qml.DepolarizingChannel(p, wires=wire) for wire in wires]

# noise_channel_2_qubits = lambda p,wires: qml.QubitChannel(pyquil.noise.pauli_kraus_map([1-p]+[p/15]*15), wires=wires)

# #noise_channel = lambda p, wires: [TwoQubitDepolarizingChannel(p, wires=[0,1]) for wire in wires]
# dev = qml.device("default.mixed", wires=num_wires, shots=shots_device, analytic=analytic_device)
# k = noisy_kernel(
#     rigetti_ansatz_mapped,
#     dev,
#     noise_channel=[noise_channel, noise_channel_2_qubits],
#     args_noise_channel=(noise_p,),
#     noise_application_level='circuit_level',
# )
# #     k = qml.kernels.EmbeddingKernel(rigetti_ansatz_mapped, dev) # Noise-free, for testing
# k_mapped = lambda x1, x2: k(x1, x2, opt_param)

# K_raw1 = qml.kernels.square_kernel_matrix(X, k_mapped, assume_normalized_kernel=fix_diag) 
# #print(noise_channel(0,),'noise channel')

# +
from pennylane.operation import AnyWires, Channel

class TwoQubitDepolarizingChannel(Channel):
    r"""DepolarizingChannel(p, wires)
    Single-qubit symmetrically depolarizing error channel.

    This channel is modelled by the following Kraus matrices:

    .. math::
        K_0 = \sqrt{1-p} \begin{bmatrix}
                1 & 0 \\
                0 & 1
                \end{bmatrix}

    .. math::
        K_1 = \sqrt{p/3}\begin{bmatrix}
                0 & 1  \\
                1 & 0
                \end{bmatrix}

    .. math::
        K_2 = \sqrt{p/3}\begin{bmatrix}
                0 & -i \\
                i & 0
                \end{bmatrix}

    .. math::
        K_3 = \sqrt{p/3}\begin{bmatrix}
                1 & 0 \\
                0 & -1
                \end{bmatrix}

    where :math:`p \in [0, 1]` is the depolarization probability and is equally
    divided in the application of all Pauli operations.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        p (float): Each Pauli gate is applied with probability :math:`\frac{p}{3}`
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

    @classmethod
    def _kraus_matrices(cls, *params):
        p = params[0]
        K0 = np.identity(4)
        return [K0] * 16


# -
analytic_device



