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


import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
from sklearn.svm import SVC
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from pennylane_cirq import ops as cirq_ops
import nk_lib
import rsmf
formatter = rsmf.setup(r"\documentclass[twocolumn,superscriptaddress,nofootinbib]{revtex4-2}")
# -

# # Load data
# don't forget to change the file name

# +
file_name_add = 'analytic'
fun = nk_lib.closest_psd_matrix
num_wires = 5

kernel_matrices = np.load(f'kernel_matrices_{file_name_add}.npy', allow_pickle=True).numpy()

tensor_mitigated_matrices = np.load(f'mitigated_matrices_{file_name_add}.npy', allow_pickle=True)
mitigated_matrices = tensor_mitigated_matrices.numpy()

tensor_doubly_mitigated_matrices = np.load(f'doubly_mitigated_matrices_{file_name_add}_{fun.__name__}.npy', allow_pickle=True)
doubly_mitigated_matrices = tensor_doubly_mitigated_matrices.numpy()

# -

# ## Plotting mitigated noise
# In the following collection of heatmaps, the row corresponds to the mitigation technique:
#  0. No mitigation
#  1. Based on first diagonal element
#  2. Averaged over entire diagonal
#  3. split up noise per embedding (and inverse embedding) subcircuits
#
#  The columns correspond to the noise strengths given in `noise_probabilities`
#
#  The used noisy kernel (there are two for testing, namely `dummy_kernel` and `dummy_split_kernel`) is set in `calculate_kernel_matrix`.
#  
#  Note that for testing, `analytic` has to be set to `False` in the device because otherwise statistical 
#  fluctuations will prevent a perfect mitigation. 

def visualize_kernel_matrices(kernel_matrices, draw_last_cbar=False):
    num_mat = len(kernel_matrices)
    width_ratios = [1]*num_mat+[0.2]*int(draw_last_cbar)
    fig,ax = plt.subplots(1, num_mat+draw_last_cbar, figsize=(num_mat*5+draw_last_cbar, 5), gridspec_kw={'width_ratios': width_ratios})
#     sns.set()
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


# noise_probabilities = np.arange(0, 0.005, 0.001)
#print(doubly_mitigated_matrices[(0,0)])
for mats in doubly_mitigated_matrices.values():
    if len(mats)>5:
        show_mats = mats[::len(mats)//5]
    else:
        show_mats = mats
    visualize_kernel_matrices(show_mats, draw_last_cbar=True)

np.set_printoptions(precision=5)
distances = np.zeros((len(doubly_mitigated_matrices), len(kernel_matrices)))
violation = np.zeros((len(doubly_mitigated_matrices), len(kernel_matrices)))
alignment = np.zeros((len(doubly_mitigated_matrices), len(kernel_matrices)))
# for j, mat in enumerate(kernel_matrices):
#     distances[0,j] = np.linalg.norm(mat-kernel_matrices[0], 'fro')
#     violation[0,j] = np.linalg.eigvalsh(mat)[0]
norm0 = np.linalg.norm(kernel_matrices[0], 'fro')
for i, (key, mats) in enumerate(mitigated_matrices.items()):
    for j, mat in enumerate(mats):
        if mat is None:
            continue
        distances[i,j] = np.linalg.norm(mat-kernel_matrices[0], 'fro')
#         violation[i,j] = np.linalg.eigvalsh(mat)[0]
        alignment[i,j] = qml.kernels.cost_functions._matrix_inner_product(mat, kernel_matrices[0])/np.linalg.norm(mat, 'fro')/ norm0
print(distances)
print(violation)

# +
# %matplotlib notebook
fig = formatter.figure()
print(formatter.fontsize)
noise_probabilities = np.arange(0, 0.05, 0.002)
def trafo(p):
    return p

labels = ['No mitigation', 'Single diagonal element', 'Average of diagonal', 'Split channel']
markers = ['x', 'o', '.', 'd']
for i in range(4):
    probs = noise_probabilities if i==0 else trafo(noise_probabilities)
    plt.plot(probs[:len(alignment[i])], alignment[i], marker=markers[i], label=labels[i], ls='', ms=5)
plt.legend(loc='lower left', framealpha=0.8)
plt.xlabel('Base noise rate $\lambda_0$')
plt.ylabel('Alignment with exact matrix')
plt.tight_layout()
plt.savefig('../plots_and_data/device_noise_mitigation_analytic.pdf')
# -



# +
mitigated_matrices = {
    (strategy, use_entries): 
    [
    nk_lib.mitigate_global_depolarization(K, num_wires=num_wires, strategy=strategy, use_entries=use_entries)[0] 
        for K in kernel_matrices
]
    for strategy, use_entries in [(None, None), ('average', (0,)), ('average', None), ('split_channel', None)]
}

def wrap_mitigation(mat, fun=qml.kernels.displace_matrix, **kwargs):
    try:
        return fun(mat, **kwargs)
    except:
        return None

fun = nk_lib.closest_psd_matrix    
    
doubly_mitigated_matrices = {
    (strategy, use_entries): 
    [
    wrap_mitigation(
        nk_lib.mitigate_global_depolarization(K, num_wires=num_wires, strategy=strategy, use_entries=use_entries)[0],
        fun=fun,
        fix_diag=True,
    ) 
        for K in kernel_matrices
]
    for strategy, use_entries in [(None, None), ('average', (0,)), ('average', None), ('split_channel', None)]
}
# -




