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
# -

# # Load data
# don't forget to change the file name

tensor_doubly_mitigated_matrices = np.load('doubly_mitigated_matrices.npy', allow_pickle=True)
doubly_mitigated_matrices = tensor_doubly_mitigated_matrices.numpy()
kernel_matrices = np.load('kernel_matrices.npy', allow_pickle=True).numpy()


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

def visualize_kernel_matrices(kernel_matrices, noise_probabilites, draw_last_cbar=False):
    num_mat = 4
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


noise_probabilities = np.arange(0, 0.005, 0.001)
#print(doubly_mitigated_matrices[(0,0)])
for mats in doubly_mitigated_matrices.values():
    visualize_kernel_matrices(mats, noise_probabilities, draw_last_cbar=True)

# +
np.set_printoptions(precision=5)
distances = np.zeros((len(doubly_mitigated_matrices), len(kernel_matrices)))
violation = np.zeros((len(doubly_mitigated_matrices), len(kernel_matrices)))

# for j, mat in enumerate(kernel_matrices):
#     distances[0,j] = np.linalg.norm(mat-kernel_matrices[0], 'fro')
#     violation[0,j] = np.linalg.eigvalsh(mat)[0]
for i, (key, mats) in enumerate(doubly_mitigated_matrices.items()):
    for j, mat in enumerate(mats):
        distances[i,j] = np.linalg.norm(mat-kernel_matrices[0], 'fro')
        violation[i,j] = np.linalg.eigvalsh(mat)[0]
print(distances)
print(violation)
# -

plt.plot(noise_probabilities, distances[0])
plt.plot(noise_probabilities, distances[1])
plt.plot(noise_probabilities, distances[2])
plt.plot(noise_probabilities, distances[3])


