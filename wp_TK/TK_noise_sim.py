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

# # Ansatz

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
        
# We use fixed "random" parameters for the Hardware runs and noise simulation
# def random_params(num_wires, num_layers):
#     return np.random.uniform(0, 2*np.pi, (num_layers, 2, num_wires))


# -

# # Tiny stuff

total_executions = 0
wires = list(range(5))

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


# Create a new data frame. WARNING, if you already computed something on the quantum computer, 
# this might reset your data if it is only stored in df! 
#
#                                             WARNING: Read the above
#
df = pd.DataFrame()

# # Run and validate classifications - SIMULATION

# +
# This is how often the entire task is going to be repeated, for statistics!
n_noises = 300
noise_levels = [1., 0.]
# These are the keywords for which kernel parameters to run the entire thing:
param_names = ['Zero', 'Random', 'Optimal']

alignments = [k.target_alignment(X, Y, _par) for _par in [zero_param, rnd_param, opt_param]]

for noise_level in noise_levels:
    for name_params, init_params, TA in zip(param_names, [zero_param, rnd_param, opt_param], alignments):
        K_raw = k.square_kernel_matrix(X, init_params) # Use quantum computer
        for i in range(n_noises if noise_level>0. else 1):
            N_train = np.random.normal(scale=noise_level, size=(len(X), len(X)))
            N_train = np.triu(N_train, 1) + np.triu(N_train, 1).T
            N_test = np.random.normal(scale=noise_level, size=(len(X), len(X)))
            N_test = np.triu(N_test, 1) + np.triu(N_test, 1).T
            for name_stabilize, stabilize in zip(['None', 'Thresholding', 'Displacing'], [None, qml.kernels.threshold_matrix, qml.kernels.displace_matrix]):
#                 print(init_params)
                kernel_mat1 = lambda A, B: stabilize(K_raw+N_train) if stabilize is not None else K_raw+N_train
                kernel_mat2 = lambda A, B: stabilize(K_raw+N_test) if stabilize is not None else K_raw+N_test
                
                svm = SVC(kernel=kernel_mat1).fit(X, Y)
                perf_reuse = tk.validate(svm, X, Y)
                svm.kernel = kernel_mat2
                perf_recompute = tk.validate(svm, X, Y)
            
                entry = pd.Series({
                    'noise_level': noise_level,
                    'params': name_params,
                    'params_long': name_params+f"\n(Alignment={TA})",
                    'noise_iteration': i,
                    'Stabilisation method': name_stabilize,
                    'perf_reuse': perf_reuse,
                    'perf_recompute': perf_recompute,
                    'alignment': TA,
                })
                df = df.append(entry, ignore_index=True)
# -

# # Run and validate classifications - QUANTUM DEVICE

# +
# This is how often the entire task is going to be repeated, for statistics! -> TO be discussed
n_repeat = 1 
# This should be replace by the different number of shots per measurement value that we want to use, say [10, 100]
shots_list = [10, 100]
# These are the keywords for which kernel parameters to run the entire thing:
# param_names = ['Zero', 'Random', 'Optimal'] # CHANGE HERE TO ADD OTHER PARAMETER RUNS
param_names = ['Optimal']

recompute_K_for_testing = False

for shots in shots_list:
    # use the following two lines when running on a QC and add the shots parameter to the device initialization!
    total_executions += dev.num_executions
    if local:
        print("You're safe")
        dev = qml.device("default.qubit", wires=5, shots=shots)
    else:
        print("MONEY IS BEING USED")
        bucket = "KEY" # the name of the bucket
        prefix = "example_running_quantum_circuits_on_qpu_devices" # the name of the folder in the bucket
        s3_folder = (bucket, prefix)
        dev_arn = "arn:aws:braket:::device/qpu/rigetti/Aspen-9"
        # final safeguard: remove the comment
        #dev = qml.device("braket.aws.qubit", device_arn=dev_arn, s3_destination_folder=s3_folder, wires=5, shots=shots, parallel=True)

    
    
    k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x @ W, params, wires), dev)
    for name_params, init_params in zip(param_names, [zero_param, rnd_param, opt_param]):
        K_raw1 = k.square_kernel_matrix(X, init_params) # Use quantum computer
        if recompute_K_for_testing:
            K_raw2 = k.square_kernel_matrix(X, init_params) # Use quantum computer
        perf_this_params = {'None': [], 'Thresholding': [], 'Displacing': []}
        for i in range(n_repeat):
            for name_stabilize, stabilize in zip(['None', 'Thresholding', 'Displacing'], [None, qml.kernels.threshold_matrix, qml.kernels.displace_matrix]):
#                 print(init_params)
                kernel_mat1 = lambda A, B: stabilize(K_raw1) if stabilize is not None else K_raw1
                if recompute_K_for_testing:
                    kernel_mat2 = lambda A, B: stabilize(K_raw2) if stabilize is not None else K_raw2
                                
                svm = SVC(kernel=kernel_mat1).fit(X, Y)
                perf_reuse = tk.validate(svm, X, Y)
                if recompute_K_for_testing:
                    svm.kernel = kernel_mat2
                    perf_recompute = tk.validate(svm, X, Y)
                else:
                    perf_recompute = None
                entry = pd.Series({
                    'shots': shots,
                    'params': name_params,
                    'noise_iteration': i,
                    'Stabilisation method': name_stabilize,
                    'perf_reuse': perf_reuse,
                    'perf_recompute': perf_recompute,
                })
                df = df.append(entry, ignore_index=True)
                
# -

# # Plot classification performances

# +
# %matplotlib notebook
import seaborn as sns
sns.set_theme(style="whitegrid")

# plot_df = df.loc[df.shots==100]
plot_df = df.loc[df.noise_level==1.]

# print(plot_df)
g = sns.catplot(
    data=plot_df, kind="bar",
    x="params", y="perf_recompute", hue='Stabilisation method',
    ci='sd', palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Classification performance on training set")
# plt.gca().set_title(f"5 Qubits, 6 layers, {n_noises} kernel matrix instances")
# g.legend.set_title("")
# plt.tight_layout()
# -
print("Total executions (check if correct by hand):", total_executions)


# +
# UNiQ = np.copy(params)
# use_UNiQ = np.copy(UNiQ)

# +
# init_params = np.copy(use_UNiQ)
# noise_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# noise_levels = [1.]

# def kernel_matrix(A, B, noise_level, stabilize=None, seed=None):
    
#     np.random.seed(seed if seed is not None else np.random.randint(0,10000))
#     if A.shape==B.shape and np.allclose(A, B):
#         N_ = np.random.normal(scale=noise_level, size=(len(A), len(A)))
#         N = np.triu(N_, 1) + np.triu(N_, 1).T
#         K = k.square_kernel_matrix(A, init_params)+N
#         spectrum = np.linalg.eigvalsh(K)
#         print(f"Smallest eigenvalue at noise level {noise_level}: {np.min(spectrum)}")
#         if stabilize=='threshold':
#             K = qml.kernels.threshold_matrix(K)
#         elif stabilize=='displace':
#             K = qml.kernels.displace_matrix(K)
#     else:
#         N = np.random.normal(scale=noise_level, size=(len(A), len(B)))
#         K = k.kernel_matrix(A, B, init_params)+N 
        
#     return K
# # kernel_matrix(X, X, noise_levels[0])

# performances = []
# for noise_level in noise_levels:
#     seed = np.random.randint(0,10000) # Fix seed for all stabilization methods.
#     perf_ = []
#     for stabilize in [None, 'threshold', 'displace']:
#         svm = SVC(kernel=lambda A, B: kernel_matrix(A, B, noise_level, stabilize, seed)).fit(X, Y)
#         perf = tk.validate(svm, X, Y)
#         print(f"For noise level {noise_level} with stabilize={stabilize} the performance is {perf}")
#         perf_.append(perf)
#     performances.append(perf_)

# -





