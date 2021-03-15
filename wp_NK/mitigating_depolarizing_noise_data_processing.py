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


import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
from sklearn.svm import SVC
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from pennylane_cirq import ops as cirq_ops
import nk_lib
# import rsmf
from dill import load
# formatter = rsmf.setup(r"\documentclass[twocolumn,superscriptaddress,nofootinbib]{revtex4-2}")
# -

# # Load raw kernel matrices

# +
filename = 'data/kernel_matrices_2d_sweep_more_noise.dill'
# fun = nk_lib.closest_psd_matrix
num_wires = 5

kernel_matrices = load(open(filename, 'rb+'))
# kernel_matrices = {}

previous_filename = 'data/kernel_matrices_2d_sweep.dill'
previous_kernel_matrices = load(open(previous_filename, 'rb+'))

kernel_matrices = {**previous_kernel_matrices, **kernel_matrices}

# tensor_mitigated_matrices = np.load(f'mitigated_matrices_{file_name_add}.npy', allow_pickle=True)
# mitigated_matrices = tensor_mitigated_matrices.numpy()

# tensor_doubly_mitigated_matrices = np.load(f'doubly_mitigated_matrices_{file_name_add}_{fun.__name__}.npy', allow_pickle=True)
# doubly_mitigated_matrices = tensor_doubly_mitigated_matrices.numpy()

# -

# # Apply mitigation techniques

# +
pipelines = {
    'None': [],
    'sdp': [(nk_lib.closest_psd_matrix, {'fix_diagonal': True})],
    'displace': [(qml.kernels.postprocessing.displace_matrix, {})],
    'thresh': [(qml.kernels.postprocessing.threshold_matrix, {})],
    'single': [(nk_lib.mitigate_global_depolarization, {'num_wires': num_wires, 'strategy': 'average', 'use_entries': (0,)})],
    'avg': [(nk_lib.mitigate_global_depolarization, {'num_wires': num_wires, 'strategy': 'average', 'use_entries': None})],
    'avg_sdp': [
        (nk_lib.mitigate_global_depolarization, {'num_wires': num_wires, 'strategy': 'average'}),
        (nk_lib.closest_psd_matrix, {'fix_diagonal': True}),
    ],
    'displace_avg_sdp': [
        (qml.kernels.postprocessing.displace_matrix, {}),
        (nk_lib.mitigate_global_depolarization, {'num_wires': num_wires, 'strategy': 'average'}),
        (nk_lib.closest_psd_matrix, {'fix_diagonal': True}),
    ],
    'displace_avg': [
        (qml.kernels.postprocessing.displace_matrix, {}),
        (nk_lib.mitigate_global_depolarization, {'num_wires': num_wires, 'strategy': 'average'}),
    ],
    'split': [(nk_lib.mitigate_global_depolarization, {'num_wires': num_wires, 'strategy': 'split_channel'})], 
    'split_sdp': [
        (nk_lib.mitigate_global_depolarization, {'num_wires': num_wires, 'strategy': 'split_channel'}),
        (nk_lib.closest_psd_matrix, {'fix_diagonal': True}),
    ],
    'displace_split_sdp': [
        (qml.kernels.postprocessing.displace_matrix, {}),
        (nk_lib.mitigate_global_depolarization, {'num_wires': num_wires, 'strategy': 'split_channel'}),
        (nk_lib.closest_psd_matrix, {'fix_diagonal': True}),
    ],

}

df = pd.DataFrame()

mitigated_matrices = {}
mitigated_alignment = {}
exact_matrix = kernel_matrices[(0., 0)]
norm0 = np.linalg.norm(exact_matrix, 'fro')
print(norm0)

for pipeline_name, pipeline in pipelines.items():
    mitigated = {}
    alignment = {}
    for key, mat in kernel_matrices.items():
        K = np.copy(mat)
        for fun, kwargs in pipeline:
            try:
                K = fun(K, **kwargs)
                if np.any(np.isinf(K)):
                    raise ValueError
            except Exception as e:
#                 print(e)
                K = None
                align = np.nan
                break
        else:
            mat_ip = qml.kernels.cost_functions._matrix_inner_product(K, exact_matrix)
            normK = np.linalg.norm(K, 'fro')
            if np.isclose(normK, 0.):
                align = np.nan
            else:
                align = mat_ip / (normK * norm0)
        alignment[key] = align
        mitigated[key] = K
        df = df.append(pd.Series(
            {
                    'base_noise_rate': key[0],
                    'shots': key[1],
                    'pipeline': pipeline_name,
#                     'mitigated_kernel_matrix': K,
                    'alignment': np.real(align),
                    'shots_sort': key[1] if key[1]>0 else int(1e10),
                }),
            ignore_index=True,
            )
    mitigated_matrices[pipeline_name] = mitigated
    mitigated_alignment[pipeline_name] = alignment

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


# +

# shots = 0

# for key, mats in mitigated_matrices.items():
# #     print(key,mats)
#     show_mats = [mat for k, mat in sorted(list(mats.items()), key=lambda x: x[0][0]) if shots in k]
#     if len(mats)>5:
#         show_mats = show_mats[::len(show_mats)//5]
#     else:
#         show_mats = show_mats
# #     print(show_mats)
# #     show_mats = [mat for mat in show_mats if mat is not None]
#     nk_lib.visualize_kernel_matrices(show_mats, draw_last_cbar=True)

# +
# np.set_printoptions(precision=5)
# distances = np.zeros((len(doubly_mitigated_matrices), len(kernel_matrices)))
# violation = np.zeros((len(doubly_mitigated_matrices), len(kernel_matrices)))
# alignment = np.zeros((len(doubly_mitigated_matrices), len(kernel_matrices)))
# # for j, mat in enumerate(kernel_matrices):
# #     distances[0,j] = np.linalg.norm(mat-kernel_matrices[0], 'fro')
# #     violation[0,j] = np.linalg.eigvalsh(mat)[0]
# norm0 = np.linalg.norm(kernel_matrices[0], 'fro')
# for i, (key, mats) in enumerate(mitigated_matrices.items()):
#     for j, mat in enumerate(mats):
#         if mat is None:
#             continue
#         distances[i,j] = np.linalg.norm(mat-kernel_matrices[0], 'fro')
# #         violation[i,j] = np.linalg.eigvalsh(mat)[0]
#         alignment[i,j] = qml.kernels.cost_functions._matrix_inner_product(mat, kernel_matrices[0])/np.linalg.norm(mat, 'fro')/ norm0
# print(distances)
# print(violation)
# -

df

# +
# # %matplotlib notebook
# fig = formatter.figure()
# print(formatter.fontsize)
# noise_probabilities = np.arange(0, 0.05, 0.002)
# def trafo(p):
#     return p

# labels = ['No mitigation', 'Single diagonal element', 'Average of diagonal', 'Split channel']
# markers = ['x', 'o', '.', 'd']
# for i in range(4):
#     probs = noise_probabilities if i==0 else trafo(noise_probabilities)
#     plt.plot(probs[:len(alignment[i])], alignment[i], marker=markers[i], label=labels[i], ls='', ms=5)
# plt.legend(loc='lower left', framealpha=0.8)
# plt.xlabel('Base noise rate $\lambda_0$')
# plt.ylabel('Alignment with exact matrix')
# plt.tight_layout()
# plt.savefig('../plots_and_data/device_noise_mitigation_analytic.pdf')
# +
# %matplotlib notebook
figsize = (9, 3*len(pipelines))
fig, ax = plt.subplots(len(pipelines), 1, figsize=figsize)
titles = {
    'None': "No Postprocessing",
    'sdp': "Best Regularization", # SDP is best of the three regularization methods alone.
    'displace': "Displacing", # Worse average than SDP alone
    'thresh': "Thresholding", # Worse worst output than SDP alone
    'single': "Single Rate Mitigation (Single)",
    'avg': "Single Rate Mitigation (Average)",
    'avg_sdp': "Single Rate Mitigation (Average) and Regularization",
    'displace_avg_sdp': "Displacing, Single Rate Mitigation (Average) and Regularization",
    'displace_avg': "Displacing and Single Rate Mitigation (Average)",
    'split': "Feature-dependent Rate Mitigation",
    'split_sdp': "Feature-dependent Rate Mitigation and Regularization",
    'displace_split_sdp': "Displacing, Feature-dependent Rate Mitigation and Regularization",
}

print(f"Worst performances:")
for i, pipeline_name in enumerate(pipelines.keys()):
    subdf = df.loc[df['pipeline']==pipeline_name]
    subdf_pivot = subdf.pivot('shots_sort', 'base_noise_rate', 'alignment')
    
    min_alignment = np.min(subdf['alignment'])
    min_alignment_finite = np.min(subdf.loc[subdf['shots']>0]['alignment'])
    min_df = subdf.loc[[subdf['alignment'].idxmin()]]
    min_finite_df = subdf.loc[[subdf.loc[subdf['shots']>0]['alignment'].idxmin()]]
    plot = sns.heatmap(data=subdf_pivot,
                vmin=-1,
                vmax=1+1e-5,
                cbar=True,
                ax=ax[i],
                linewidth=0.2,
#                 xticklabels=list(df['base_noise_rate'].unique()[::5]),
#                 xticks=list(df['base_noise_rate'].unique()[::5]),
                yticklabels=list(np.round(df['shots_sort'].astype(int).unique()[:-1],0))+['analytic'],
               )
    if i<len(pipelines)-1:
        ax[i].set_xticks([])
        ax[i].set_xlabel('')
    else:
        
        print(ax[i].get_xticklabels())
        ax[i].set_xticklabels([ticklabel.get_text()[:4] for ticklabel in ax[i].get_xticklabels()])
        ax[i].set_xticks(ax[i].get_xticks()[::5])
        plt.setp( ax[i].xaxis.get_majorticklabels(), rotation=0 )
        ax[i].set_xlabel('Base noise rate $\\lambda_0$')
    plt.setp( ax[i].yaxis.get_majorticklabels(), rotation=0 )
    ax[i].set_ylabel('# Measurements')
    ax[i].set_title(titles[pipeline_name])
    
    cbar = ax[i].collections[0].colorbar
    # Tick 1
    tick_col = 'k' if min_alignment > 0.2 else '1'
    cbar.ax.hlines(min_alignment, -1.2, 1.2, color=tick_col)
    cbar.ax.text(-1.5, min_alignment, f"{min_alignment:.2f}", horizontalalignment='right', verticalalignment='center')
    shot_coord = np.log10(min_df['shots_sort'].item()) if min_df['shots_sort'].item() <1e6 else 5
    ax[i].plot((min_df['base_noise_rate'].item()/0.002)+0.5, shot_coord-0.5, marker='x', color=tick_col)
    if min_df['shots'].item() == 0:
        # Tick 2
        tick_col = 'k' if min_alignment_finite > 0.2 else '1'
        cbar.ax.hlines(min_alignment_finite, -1.2, 1.2, color=tick_col)
        cbar.ax.text(-1.5, min_alignment_finite, f"{min_alignment_finite:.2f}", horizontalalignment='right', verticalalignment='center')
        shot_coord = np.log10(min_finite_df['shots_sort'].item()) if min_finite_df['shots_sort'].item() <1e6 else 5
        ax[i].plot((min_finite_df['base_noise_rate'].item()/0.002)+0.5, shot_coord-0.5, marker='x', color=tick_col)
#     ax[i].text(0.0, 4.5, f'Minimal alignment: {min_alignment:.3f}')
#     print(np.min(subdf['alignment']), np.max(subdf['alignment']))

    print(f"{titles[pipeline_name]} - {min_alignment}")
plt.tight_layout()
plt.savefig('2d_sweep_mitigation_and_regularization_more_noise.pdf')
# -


df['base_noise_rate']

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




