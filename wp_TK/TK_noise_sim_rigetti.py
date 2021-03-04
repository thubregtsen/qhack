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

# # Dataset

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


# # Run and validate classifications - QUANTUM DEVICE

# +
# This is how often the entire task is going to be repeated, for statistics! -> TO be discussed
n_repeat = 1 
# This should be replace by the different number of shots per measurement value that we want to use, say [10, 100]
#shots_list = [10, 100]
# These are the keywords for which kernel parameters to run the entire thing:


total_executions = 0
recompute_K_for_testing = False
shots = 100


local = False
if local == True:
    print("You're safe")
    dev = qml.device("default.qubit", wires=5, shots=shots, analytic=False)
else:
    print("MONEY IS BEING USED")
    bucket = "amazon-braket-rigetti-fub" # the name of the bucket
    prefix = "noisy_sim" # the name of the folder in the bucket
    s3_folder = (bucket, prefix)
    dev_arn = "arn:aws:braket:::device/qpu/rigetti/Aspen-9"
    # final safeguard: remove the comment
    dev = qml.device("braket.aws.qubit", device_arn=dev_arn, s3_destination_folder=s3_folder, wires=5, shots=100, parallel=True)

k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x @ W, params, wires), dev)

#for name_params, init_params in zip(param_names, [zero_param, rnd_param, opt_param]):
name_params = 'optimal'
init_params = opt_param


K_raw1 = k.square_kernel_matrix(X, init_params) # Use quantum computer
perf_this_params = {'None': [], 'Thresholding': [], 'Displacing': []}
for i in range(n_repeat):
    for name_stabilize, stabilize in zip(['None', 'Thresholding', 'Displacing'], [None, qml.kernels.threshold_matrix, qml.kernels.displace_matrix]):
        kernel_mat1 = lambda A, B: stabilize(K_raw1) if stabilize is not None else K_raw1
        if recompute_K_for_testing:
            kernel_mat2 = lambda A, B: stabilize(K_raw2) if stabilize is not None else K_raw2

        svm = SVC(kernel=kernel_mat1).fit(X, Y)
        perf_reuse = tk.validate(svm, X, Y)
        if recompute_K_for_testing:
            svm.kernel = kernel_mat2
            perf_recompute = tk.validate(svm, X, Y)
        else:
            perf_recompute = 0
        entry = pd.Series({
            'shots': shots,
            'params': name_params,
            'noise_iteration': i,
            'Stabilisation method': name_stabilize,
            'perf_reuse': perf_reuse,
            'perf_recompute': perf_recompute,
        })
        df = df.append(entry, ignore_index=True)

total_executions += dev.num_executions

print(total_executions)
# -

# # Plot classification performances

# +
# %matplotlib notebook
import seaborn as sns
sns.set_theme(style="whitegrid")

plot_df = df.loc[df.shots==100]
#plot_df = df.loc[df.noise_level==1.]

# print(plot_df)
g = sns.catplot(
    data=plot_df, kind="bar",
    x="params", y="perf_reuse", hue='Stabilisation method',
    ci='sd', palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Classification performance on training set")
# plt.gca().set_title(f"5 Qubits, 6 layers, {n_noises} kernel matrix instances")
# g.legend.set_title("")
# plt.tight_layout()
# -
print("Total executions (check if correct by hand):", total_executions)


print(df)

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
for name_params, init_params in zip(param_names, [zero_param, rnd_param, opt_param]):
    print(name_params, init_params)

K_raw1

# +
kernel_matrix_from_rigetti=np.tensor([[1.  , 0.03, 0.02, 0.  , 0.01, 0.02, 0.01, 0.  , 0.02, 0.01,
         0.02, 0.  ],
        [0.03, 1.  , 0.01, 0.01, 0.02, 0.01, 0.  , 0.  , 0.01, 0.  ,
         0.01, 0.01],
        [0.02, 0.01, 1.  , 0.  , 0.  , 0.03, 0.01, 0.03, 0.  , 0.02,
         0.02, 0.01],
        [0.  , 0.01, 0.  , 1.  , 0.01, 0.01, 0.  , 0.  , 0.02, 0.01,
         0.03, 0.02],
        [0.01, 0.02, 0.  , 0.01, 1.  , 0.01, 0.01, 0.  , 0.  , 0.02,
         0.04, 0.01],
        [0.02, 0.01, 0.03, 0.01, 0.01, 1.  , 0.  , 0.  , 0.01, 0.01,
         0.03, 0.  ],
        [0.01, 0.  , 0.01, 0.  , 0.01, 0.  , 1.  , 0.01, 0.  , 0.01,
         0.  , 0.01],
        [0.  , 0.  , 0.03, 0.  , 0.  , 0.  , 0.01, 1.  , 0.01, 0.  ,
         0.  , 0.03],
        [0.02, 0.01, 0.  , 0.02, 0.  , 0.01, 0.  , 0.01, 1.  , 0.03,
         0.  , 0.  ],
        [0.01, 0.  , 0.02, 0.01, 0.02, 0.01, 0.01, 0.  , 0.03, 1.  ,
         0.01, 0.02],
        [0.02, 0.01, 0.02, 0.03, 0.04, 0.03, 0.  , 0.  , 0.  , 0.01,
         1.  , 0.  ],
        [0.  , 0.01, 0.01, 0.02, 0.01, 0.  , 0.01, 0.03, 0.  , 0.02,
         0.  , 1.  ]], requires_grad=True)


noiseless_simulation = np.array([[1.   0.07 0.06 0.   0.28 0.03 0.2  0.34 0.03 0.16 0.07 0.16]
 [0.07 1.   0.04 0.34 0.06 0.12 0.24 0.   0.31 0.02 0.35 0.02]
 [0.06 0.04 1.   0.05 0.12 0.07 0.24 0.46 0.61 0.07 0.19 0.06]
 [0.   0.34 0.05 1.   0.01 0.1  0.23 0.05 0.21 0.28 0.33 0.03]
 [0.28 0.06 0.12 0.01 1.   0.03 0.08 0.31 0.01 0.21 0.03 0.29]
 [0.03 0.12 0.07 0.1  0.03 1.   0.15 0.2  0.16 0.04 0.38 0.4 ]
 [0.2  0.24 0.24 0.23 0.08 0.15 1.   0.53 0.41 0.08 0.27 0.02]
 [0.34 0.   0.46 0.05 0.31 0.2  0.53 1.   0.35 0.28 0.1  0.24]
 [0.03 0.31 0.61 0.21 0.01 0.16 0.41 0.35 1.   0.   0.14 0.02]
 [0.16 0.02 0.07 0.28 0.21 0.04 0.08 0.28 0.   1.   0.55 0.42]
 [0.07 0.35 0.19 0.33 0.03 0.38 0.27 0.1  0.14 0.55 1.   0.23]
 [0.16 0.02 0.06 0.03 0.29 0.4  0.02 0.24 0.02 0.42 0.23 1.  ]])

# -

from sklearn.metrics import plot_confusion_matrix

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# +
# %matplotlib notebook

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
kernel_matrix_from_rigetti=np.array([[1.  , 0.03, 0.02, 0.  , 0.01, 0.02, 0.01, 0.  , 0.02, 0.01,
         0.02, 0.  ],
        [0.03, 1.  , 0.01, 0.01, 0.02, 0.01, 0.  , 0.  , 0.01, 0.  ,
         0.01, 0.01],
        [0.02, 0.01, 1.  , 0.  , 0.  , 0.03, 0.01, 0.03, 0.  , 0.02,
         0.02, 0.01],
        [0.  , 0.01, 0.  , 1.  , 0.01, 0.01, 0.  , 0.  , 0.02, 0.01,
         0.03, 0.02],
        [0.01, 0.02, 0.  , 0.01, 1.  , 0.01, 0.01, 0.  , 0.  , 0.02,
         0.04, 0.01],
        [0.02, 0.01, 0.03, 0.01, 0.01, 1.  , 0.  , 0.  , 0.01, 0.01,
         0.03, 0.  ],
        [0.01, 0.  , 0.01, 0.  , 0.01, 0.  , 1.  , 0.01, 0.  , 0.01,
         0.  , 0.01],
        [0.  , 0.  , 0.03, 0.  , 0.  , 0.  , 0.01, 1.  , 0.01, 0.  ,
         0.  , 0.03],
        [0.02, 0.01, 0.  , 0.02, 0.  , 0.01, 0.  , 0.01, 1.  , 0.03,
         0.  , 0.  ],
        [0.01, 0.  , 0.02, 0.01, 0.02, 0.01, 0.01, 0.  , 0.03, 1.  ,
         0.01, 0.02],
        [0.02, 0.01, 0.02, 0.03, 0.04, 0.03, 0.  , 0.  , 0.  , 0.01,
         1.  , 0.  ],
        [0.  , 0.01, 0.01, 0.02, 0.01, 0.  , 0.01, 0.03, 0.  , 0.02,
         0.  , 1.  ]])


kernel_matrix_from_simulation = np.array([[1.  , 0.03, 0.05, 0.  , 0.23, 0.1 , 0.26, 0.39, 0.  , 0.3 ,
         0.06, 0.15],
        [0.03, 1.  , 0.07, 0.22, 0.03, 0.13, 0.18, 0.  , 0.29, 0.02,
         0.38, 0.02],
        [0.05, 0.07, 1.  , 0.05, 0.07, 0.05, 0.23, 0.42, 0.53, 0.07,
         0.19, 0.13],
        [0.  , 0.22, 0.05, 1.  , 0.  , 0.07, 0.26, 0.05, 0.11, 0.28,
         0.38, 0.03],
        [0.23, 0.03, 0.07, 0.  , 1.  , 0.04, 0.02, 0.29, 0.03, 0.17,
         0.01, 0.15],
        [0.1 , 0.13, 0.05, 0.07, 0.04, 1.  , 0.07, 0.14, 0.14, 0.07,
         0.34, 0.45],
        [0.26, 0.18, 0.23, 0.26, 0.02, 0.07, 1.  , 0.47, 0.36, 0.15,
         0.25, 0.  ],
        [0.39, 0.  , 0.42, 0.05, 0.29, 0.14, 0.47, 1.  , 0.23, 0.3 ,
         0.09, 0.23],
        [0.  , 0.29, 0.53, 0.11, 0.03, 0.14, 0.36, 0.23, 1.  , 0.01,
         0.23, 0.02],
        [0.3 , 0.02, 0.07, 0.28, 0.17, 0.07, 0.15, 0.3 , 0.01, 1.  ,
         0.49, 0.39],
        [0.06, 0.38, 0.19, 0.38, 0.01, 0.34, 0.25, 0.09, 0.23, 0.49,
         1.  , 0.31],
        [0.15, 0.02, 0.13, 0.03, 0.15, 0.45, 0.  , 0.23, 0.02, 0.39,
         0.31, 1.  ]])

matrix_from_ionq  = np.array([[1.  , 0.02, 0.01, 0.04, 0.02, 0.06, 0.01, 0.  , 0.04, 0.02,
         0.  , 0.03],
        [0.02, 1.  , 0.02, 0.08, 0.  , 0.03, 0.03, 0.02, 0.02, 0.08,
         0.02, 0.04],
        [0.01, 0.02, 1.  , 0.  , 0.03, 0.02, 0.06, 0.05, 0.04, 0.04,
         0.04, 0.03],
        [0.04, 0.08, 0.  , 1.  , 0.02, 0.06, 0.03, 0.06, 0.05, 0.03,
         0.05, 0.02],
        [0.02, 0.  , 0.03, 0.02, 1.  , 0.03, 0.06, 0.08, 0.01, 0.05,
         0.09, 0.03],
        [0.06, 0.03, 0.02, 0.06, 0.03, 1.  , 0.01, 0.07, 0.04, 0.05,
         0.01, 0.08],
        [0.01, 0.03, 0.06, 0.03, 0.06, 0.01, 1.  , 0.05, 0.06, 0.03,
         0.05, 0.04],
        [0.  , 0.02, 0.05, 0.06, 0.08, 0.07, 0.05, 1.  , 0.05, 0.04,
         0.09, 0.06],
        [0.04, 0.02, 0.04, 0.05, 0.01, 0.04, 0.06, 0.05, 1.  , 0.05,
         0.02, 0.03],
        [0.02, 0.08, 0.04, 0.03, 0.05, 0.05, 0.03, 0.04, 0.05, 1.  ,
         0.02, 0.06],
        [0.  , 0.02, 0.04, 0.05, 0.09, 0.01, 0.05, 0.09, 0.02, 0.02,
         1.  , 0.03],
        [0.03, 0.04, 0.03, 0.02, 0.03, 0.08, 0.04, 0.06, 0.03, 0.06,
         0.03, 1.  ]])

ax, fig = plt.subplots()
sns.set()
sns.heatmap(kernel_matrix_from_simulation, vmin=0, vmax=1, xticklabels='', yticklabels='', cmap='Spectral',ax =ax1, cbar=False)
sns.heatmap(kernel_matrix_from_rigetti, vmin=0, vmax=1, cmap='Spectral', xticklabels='', yticklabels='', ax=ax2, cbar=False)
sns.heatmap(matrix_from_ionq, vmin=0, vmax=1, cmap='Spectral', xticklabels='', yticklabels='', ax=ax3)

plt.savefig('rigetti_simulation')
# -

matrix_from_ionq = np.array([[1.  , 0.03, 0.05, 0.  , 0.23, 0.1 , 0.26, 0.39, 0.  , 0.3 ,
         0.06, 0.15],
        [0.03, 1.  , 0.07, 0.22, 0.03, 0.13, 0.18, 0.  , 0.29, 0.02,
         0.38, 0.02],
        [0.05, 0.07, 1.  , 0.05, 0.07, 0.05, 0.23, 0.42, 0.53, 0.07,
         0.19, 0.13],
        [0.  , 0.22, 0.05, 1.  , 0.  , 0.07, 0.26, 0.05, 0.11, 0.28,
         0.38, 0.03],
        [0.23, 0.03, 0.07, 0.  , 1.  , 0.04, 0.02, 0.29, 0.03, 0.17,
         0.01, 0.15],
        [0.1 , 0.13, 0.05, 0.07, 0.04, 1.  , 0.07, 0.14, 0.14, 0.07,
         0.34, 0.45],
        [0.26, 0.18, 0.23, 0.26, 0.02, 0.07, 1.  , 0.47, 0.36, 0.15,
         0.25, 0.  ],
        [0.39, 0.  , 0.42, 0.05, 0.29, 0.14, 0.47, 1.  , 0.23, 0.3 ,
         0.09, 0.23],
        [0.  , 0.29, 0.53, 0.11, 0.03, 0.14, 0.36, 0.23, 1.  , 0.01,
         0.23, 0.02],
        [0.3 , 0.02, 0.07, 0.28, 0.17, 0.07, 0.15, 0.3 , 0.01, 1.  ,
         0.49, 0.39],
        [0.06, 0.38, 0.19, 0.38, 0.01, 0.34, 0.25, 0.09, 0.23, 0.49,
         1.  , 0.31],
        [0.15, 0.02, 0.13, 0.03, 0.15, 0.45, 0.  , 0.23, 0.02, 0.39,
         0.31, 1.  ]])



# +
matrix_from_ionq  = np.array([[1.  , 0.02, 0.01, 0.04, 0.02, 0.06, 0.01, 0.  , 0.04, 0.02,
         0.  , 0.03],
        [0.02, 1.  , 0.02, 0.08, 0.  , 0.03, 0.03, 0.02, 0.02, 0.08,
         0.02, 0.04],
        [0.01, 0.02, 1.  , 0.  , 0.03, 0.02, 0.06, 0.05, 0.04, 0.04,
         0.04, 0.03],
        [0.04, 0.08, 0.  , 1.  , 0.02, 0.06, 0.03, 0.06, 0.05, 0.03,
         0.05, 0.02],
        [0.02, 0.  , 0.03, 0.02, 1.  , 0.03, 0.06, 0.08, 0.01, 0.05,
         0.09, 0.03],
        [0.06, 0.03, 0.02, 0.06, 0.03, 1.  , 0.01, 0.07, 0.04, 0.05,
         0.01, 0.08],
        [0.01, 0.03, 0.06, 0.03, 0.06, 0.01, 1.  , 0.05, 0.06, 0.03,
         0.05, 0.04],
        [0.  , 0.02, 0.05, 0.06, 0.08, 0.07, 0.05, 1.  , 0.05, 0.04,
         0.09, 0.06],
        [0.04, 0.02, 0.04, 0.05, 0.01, 0.04, 0.06, 0.05, 1.  , 0.05,
         0.02, 0.03],
        [0.02, 0.08, 0.04, 0.03, 0.05, 0.05, 0.03, 0.04, 0.05, 1.  ,
         0.02, 0.06],
        [0.  , 0.02, 0.04, 0.05, 0.09, 0.01, 0.05, 0.09, 0.02, 0.02,
         1.  , 0.03],
        [0.03, 0.04, 0.03, 0.02, 0.03, 0.08, 0.04, 0.06, 0.03, 0.06,
         0.03, 1.  ]])
sns.set()

sns.heatmap(matrix_from_ionq, vmin=0, vmax=1, cmap='Spectral', xticklabels='', yticklabels='', ax=ax2)
plt.show()
# -


