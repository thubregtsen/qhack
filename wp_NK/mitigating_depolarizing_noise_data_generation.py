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

# # Simulate noise

def add_noise_channel(operation_list, noise_channel):
    """This is not tested yet! A less hacky version to add noise to the circuit, copied from qml.inv """
    if isinstance(operation_list, qml.operation.Operation):
        operation_list = [operation_list]
    elif operation_list is None:
        raise ValueError(
            "None was passed as an argument to add_noise_channel. "
            "This could happen if adding noise to a template without the template decorator is attempted."
        )
    elif callable(operation_list):
        raise ValueError(
            "A function was passed as an argument to add_noise_channel. "
            "This could happen if adding noise to a template function is attempted. "
            "Please use add_noise_channel on the function including its arguments, "
            "as in add_noise_channel(template(args), channel)."
        )
#     elif isinstance(operation_list, qml.tape.QuantumTape):
#         operation_list.inv()
#         return operation_list
    elif not isinstance(operation_list, Iterable):
        raise ValueError("The provided operation_list is not iterable.")

    non_ops = [
        (idx, op)
        for idx, op in enumerate(operation_list)
        if not isinstance(op, qml.operation.Operation)
    ]

    if non_ops:
        string_reps = [" operation_list[{}] = {}".format(idx, op) for idx, op in non_ops]
        raise ValueError(
            "The given operation_list does not only contain Operations."
            + "The following elements of the iterable were not Operations:"
            + ",".join(string_reps)
        )

    if qml.tape_mode_active():
        for op in operation_list:
            try:
                # remove the queued operation to add noise to
                # from the existing queuing context
                qml.tape.QueuingContext.remove(op)
            except KeyError:
                # operation to be inverted does not
                # exist on the queuing context
                pass

        with qml.tape.QuantumTape() as tape:
            for o in operation_list:
                o.queue()
                if o.inverse:
                    o.inv()
                (c.queue() for c in noise_channel(o.wires))

        return tape

    for op in operation_list:
        qml.QueuingContext.remove(op)
    
    all_ops = []
    for op in operation_list:
        qml.QueuingContext.append(op)
        (qml.QueuingContext.append(c) for c in noise_channel(op.wires))
        all_ops.append(op)
        (all_ops.append(c) for c in noise_channel(op.wires))

    return all_ops


# +
def apply_noise(ansatz, device, noise_channel, args_noise_channel, adjoint=False):
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
    
    @qml.template
    def noisy_ansatz(*args_ansatz, **kwargs_ansatz):
        
        # Define a circuit with which a QNode can be instantiated.
        def _ansatz(*args, **kwargs):
            if adjoint:
                qml.inv(ansatz(*args, **kwargs))
            else:
                ansatz(*args, **kwargs)
            return qml.expval(qml.PauliZ(0))

        # Instantiate and construct a qnode and extract the list of operations.
        _qnode = qml.QNode(_ansatz, _device)
        _qnode.construct(args_ansatz, kwargs_ansatz)
        ops = _qnode.qtape.operations
    
        # Apply the original operations and the noise_channel alternatingly.
        for op in ops:
            if op.inverse:
                op.__class__(*op.data, wires=op.wires).inv()
            else:
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
            noisy_ansatz = apply_noise(ansatz, device, noise_channel, args_noise_channel)
            noisy_adj_ansatz = apply_noise(ansatz, device, noise_channel, args_noise_channel, adjoint=True)

        def circuit(x1, x2, params, **kwargs):
            if self.noise_application_level == 'per_gate':
                noisy_ansatz(x1, params, **kwargs)
            else:
                ansatz(x1, params, **kwargs)
            if self.noise_application_level == 'per_embedding':
                self.noise_channel(*self.args_noise_channel, device.wires)
                
            if self.noise_application_level == 'per_gate':
                noisy_adj_ansatz(x2, params, **kwargs)
            else:
                qml.inv(ansatz(x2, params, **kwargs))
            if self.noise_application_level in ('per_embedding', 'global'):
                self.noise_channel(*self.args_noise_channel, device.wires)

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
# -



def mitigate_global_depolarization(kernel_matrix, num_wires, strategy='average', use_entries=None):
    """Estimate the noise rate of a global depolarizing noise model based on the diagonal entries of a kernel
    matrix and mitigate the effect of said noise model.
    Args:
      kernel_matrix (ndarray): Noisy kernel matrix.
      num_wires (int): Number of wires/qubits that was used to compute the kernel matrix.
      strategy ('average'|'split_channel'|None): Details of the noise model and strategy for mitigation.
        'average': Compute the noise rate based on the diagonal entries in use_entries, average if applicable.
        'split_channel': Assume a distinct effective noise rate for the embedding circuit of each feature vector.
        None: Don't do anything.
      use_entries (list<int>): Indices of diagonal entries to use if strategy=='average'. Set to all if None.
    Returns:
      mitigated_matrix (ndarray): Mitigated kernel matrix.
      noise_rates (ndarray): Determined noise rates, meaning depends on kwarg strategy.
    Comments:
      If strategy is 'average', the diagonal entries with indices use_entries have to be measured on the QC.
      If it is 'split_channel', all diagonal entries are required.
    """
    dim = 2**num_wires
    
    if strategy is None:
        return kernel_matrix, None
    
    elif strategy=='average':
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
noise_probabilities = np.arange(0, 0.005, 0.001)
kernel_matrices = []
fix_diag = False # Compute the diagonal entries
shots_device = 1 if shots==0 else shots # shots=0 raises an error...

rigetti_ansatz_mapped = lambda x, params: rigetti_ansatz(x @ W, params, range(num_wires))
noise_channel = lambda p, wires: [cirq_ops.Depolarize(p, wires=wire) for wire in wires]
    
for noise_p in noise_probabilities:
    dev = qml.device("cirq.mixedsimulator", wires=num_wires, shots=shots_device, analytic=analytic_device)
    print(noise_p)
    k = noisy_kernel(
        rigetti_ansatz_mapped,
        dev,
        noise_channel=noise_channel,
        args_noise_channel=(noise_p,),
        noise_application_level='per_gate',
    )
#     k = qml.kernels.EmbeddingKernel(rigetti_ansatz_mapped, dev) # Noise-free, for testing
    k_mapped = lambda x1, x2: k(x1, x2, opt_param)
    
    K_raw1 = qml.kernels.square_kernel_matrix(X, k_mapped, assume_normalized_kernel=fix_diag)    
        
    kernel_matrices.append(K_raw1)
# -


# ## Compute noise-mitigated matrices
# We look at three strategies of mitigation, corresponding to distinct assumptions on the noise.

# +
mitigated_matrices = {
    (strategy, use_entries): 
    [
    mitigate_global_depolarization(K, num_wires=num_wires, strategy=strategy, use_entries=use_entries)[0] 
        for K in kernel_matrices
]
    for strategy, use_entries in [(None, None), ('average', (0,)), ('average', None), ('split_channel', None)]
}

doubly_mitigated_matrices = {
    (strategy, use_entries): 
    [
    qml.kernels.displace_matrix(mitigate_global_depolarization(K, num_wires=num_wires, strategy=strategy, use_entries=use_entries)[0]) 
        for K in kernel_matrices
]
    for strategy, use_entries in [(None, None), ('average', (0,)), ('average', None), ('split_channel', None)]
}
# -

# # Save data

# +
mitigated_matrices_arrays = {}
for ki in doubly_mitigated_matrices.keys():
    mitigated_matrices_arrays[ki] = []
    for valu in doubly_mitigated_matrices[ki]:
        mitigated_matrices_arrays[ki].append(valu.numpy())
#print('change save file name!')
np.save('doubly_mitigated_matrices.npy', mitigated_matrices_arrays,allow_pickle=True)

np.save('kernel_matrices.npy', kernel_matrices, allow_pickle = True)
#kernel_matrices
# -


