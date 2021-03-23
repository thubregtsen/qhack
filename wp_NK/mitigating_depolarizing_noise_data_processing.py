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
# from sklearn.svm import SVC
import pandas as pd
import time

import seaborn as sns

from pennylane_cirq import ops as cirq_ops
import nk_lib
from itertools import product
# import rsmf
from dill import load, dump
import matplotlib.text as mpl_text

# formatter = rsmf.setup(r"\documentclass[twocolumn,superscriptaddress,nofootinbib]{revtex4-2}")

# +
# Helper functions for cell plotting
def get_cells(vert, horz, iterations=20):
    
    num_rows = vert.shape[0]
    num_cols = horz.shape[1]
    print(num_rows, num_cols)
    mat = np.arange(num_rows*num_cols, dtype=int).reshape((num_rows, num_cols))
    for _ in range(iterations):
        for i in range(num_rows):
            for j in range(num_cols):
                nghbhood = [(i,j)]
                if j>0 and not vert[i,j-1]:
                    nghbhood.append((i,j-1))
                if j<num_cols-1 and not vert[i,j]:
                    nghbhood.append((i,j+1))
                if i>0 and not horz[i-1,j]:
                    nghbhood.append((i-1,j))
                if i<num_rows-1 and not horz[i,j]:
                    nghbhood.append((i+1,j))
                nghb_min = np.min([mat[_i,_j] for _i,_j in nghbhood])
#                 if i==0:
#                     print(nghbhood, nghb_min)
                for _i, _j in nghbhood:
                    mat[_i,_j] = nghb_min
                    
    _map = {val: count for count, val in enumerate(np.unique(mat))}
#     print(_map)
    for i in range(num_rows):
        for j in range(num_cols):
            mat[i,j] = _map[mat[i,j]]
            
    return mat

def get_cell_centers(cells):
    ids = np.unique(cells)
    centers = {}
    for _id in ids:
        centers[_id] = np.mean(np.where(cells==_id), axis=1)
    return centers

def get_cell_label_pos(cells):
    label_pos = get_cell_centers(cells)
    ids = np.unique(cells)
    for _id in ids:
        prev_pos = [int(np.round(label_pos[_id][0])), int(np.round(label_pos[_id][1]))]
        if cells[prev_pos[0], prev_pos[1]]!= _id:
            where = np.where(cells==_id)
            center = np.mean(where, axis=1)
            dists = [(coord, np.linalg.norm(center-coord,2)) for coord in zip(where[0], where[1])]
            label_pos[_id] = min(dists, key=lambda x: x[1])[0]
        
    return label_pos



# +
# Legend handler artist to allow for Text/numeric handles, adds a circle around text

class AnyObject(object):
    def __init__(self, num, color='k'):
        self.my_text = str(num)
        self.my_color = color

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpl_text.Text(x=0, y=0, text=orig_handle.my_text, color=orig_handle.my_color, verticalalignment=u'baseline', 
                                horizontalalignment=u'left', multialignment=None, 
                                fontproperties=None, linespacing=None,
                                bbox=dict(boxstyle="circle",ec=(0., 0., 0.),fc=(1., 1., 1.), alpha=0.2),       
                                rotation_mode=None)
        handlebox.add_artist(patch)
        return patch


# -

# # Load raw kernel matrices

# +
filename = f'data/kernel_matrices_DoubleCake_untrained.dill'
filename = 'data/kernel_matrices_Checkerboard_trained.dill'

num_wires = 5

kernel_matrices = load(open(filename, 'rb+'))
# kernel_matrices = {}

# previous_filename = 'data/kernel_matrices_DoubleCake_trained_old_version.dill'
# previous_kernel_matrices = load(open(previous_filename, 'rb+'))

# kernel_matrices = {**previous_kernel_matrices, **kernel_matrices}

# dump(kernel_matrices, open(filename, 'wb+'))
# -

# # Set up pipelines for postprocessing

# +
# Set up all pipelines that make remotely sense
regularization = {
    'None': tuple(),
    'displace': (qml.kernels.postprocessing.displace_matrix, {}),
    'thresh': (qml.kernels.postprocessing.threshold_matrix, {}),
    'sdp': (nk_lib.closest_psd_matrix, {'fix_diagonal': True}),
}
mitigation = {
    'None': tuple(),
    'avg': (nk_lib.mitigate_global_depolarization, {'num_wires': num_wires, 'strategy': 'average', 'use_entries': None}),
    'single': (nk_lib.mitigate_global_depolarization, {'num_wires': num_wires, 'strategy': 'average', 'use_entries': (0,)}),
    'split': (nk_lib.mitigate_global_depolarization, {'num_wires': num_wires, 'strategy': 'split_channel'}), 
}

num_pipelines = 100

pipelines = {}
for (k1, v1), (k2, v2), (k3, v3) in product(regularization.items(), mitigation.items(),regularization.items()):
    if k1=='None':
        if k2=='None' and k3!='None':
            # Single regularization is sorted as regul_None_None, not None_None_regul
            continue
    if k2=='None':
        if k3!='None':
            # regul_None_regul does not make sense
            continue
    pipelines['_'.join([k for k in [k1, k2, k3] if k!='None'])] = [v for v in [v1, v2, v3] if v!=()]
    
    if len(pipelines) == num_pipelines:
        break
        

print(pipelines.keys())
# -

# # Apply mitigation techniques

try:
    df = pd.read_pickle(filename[:-5]+'_mitigated.dill')
except FileNotFoundError:
    df = pd.DataFrame()

    mitigated_matrices = {}
    mitigated_alignment = {}
    exact_matrix = kernel_matrices[(0., 0)]
    norm0 = np.linalg.norm(exact_matrix, 'fro')
    print(norm0)

    times_per_fun = {
        qml.kernels.postprocessing.displace_matrix: 0,
        (nk_lib.mitigate_global_depolarization, 'average'): 0,
        (nk_lib.mitigate_global_depolarization, 'single'): 0,
        (nk_lib.mitigate_global_depolarization, 'split_channel'): 0,
        nk_lib.closest_psd_matrix: 0,
        qml.kernels.postprocessing.threshold_matrix:0 ,
    }
    fun_evals = {
        qml.kernels.postprocessing.displace_matrix: 0,
        (nk_lib.mitigate_global_depolarization, 'average'): 0,
        (nk_lib.mitigate_global_depolarization, 'single'): 0,
        (nk_lib.mitigate_global_depolarization, 'split_channel'): 0,
        nk_lib.closest_psd_matrix: 0,
        qml.kernels.postprocessing.threshold_matrix:0 ,
    }

    for pipeline_name, pipeline in pipelines.items():
        print(pipeline_name)
        mitigated = {}
        alignment = {}
        for key, mat in kernel_matrices.items():
            K = np.copy(mat)
            for fun, kwargs in pipeline:

                try:
                    fun_start = time.process_time()
                    K = fun(K, **kwargs)
                    strat_str = kwargs.get('strategy', '')
                    if strat_str == 'average' and kwargs.get('use_entries', [])==(0,):
                        strat_str = 'single'
                    fun_key = fun if fun is not nk_lib.mitigate_global_depolarization else (fun, strat_str)
                    times_per_fun[fun_key] += time.process_time()-fun_start
                    fun_evals[fun_key] += 1
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
                        'base_noise_rate': np.round(key[0], 3),
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


df.reset_index(level=0, inplace=True, drop=True)
df.to_pickle(filename[:-5]+'_mitigated.dill')

df

# print(times_per_fun)
try:
    print(f"Average execution times of postprocessing functions:")
    for key, num in fun_evals.items():
        if num>0:
            print(f"{key}  -  {times_per_fun[key]/num}")
except:
    pass

# +
# Find best pipeline for each combination of shots and noise_rate

shot_numbers = sorted(list(df['shots_sort'].unique()))
noise_rates = sorted(list(df['base_noise_rate'].unique()))
all_pipelines = sorted(list(df['pipeline'].unique()))


best_alignment = np.zeros((len(shot_numbers), len(noise_rates)))
best_pipeline = np.zeros_like(best_alignment, dtype=object)
best_pipeline_id = np.zeros_like(best_alignment, dtype=int)
vert = np.zeros((len(shot_numbers), len(noise_rates)-1), dtype=bool)
horz = np.zeros((len(shot_numbers)-1, len(noise_rates)), dtype=bool)

best_df = pd.DataFrame()
for i, _shots in enumerate(shot_numbers):
    for j, _lambda in enumerate(noise_rates):
        sub_df = df.loc[(df['shots_sort']==_shots)&(df['base_noise_rate']==_lambda)]
    #     print(sub_df)
        best_sub_df = sub_df.loc[sub_df['alignment'].idxmax()]
        if best_sub_df['alignment']<sub_df['alignment'].max():
            print(sub_df['alignment'].idxmax())
            print(sub_df)
        best_df = best_df.append(best_sub_df, ignore_index=True)

        best_alignment[i, j] = best_sub_df['alignment']
        best_pipeline[i, j] = best_sub_df['pipeline']
        if j>0 and best_pipeline[i,j-1]!=best_pipeline[i,j]:
            vert[i,j-1] = True
        if i>0 and best_pipeline[i-1,j]!=best_pipeline[i,j]:
            horz[i-1,j] = True

pipeline_ids = {pipe: i for i, pipe in enumerate(sorted([pipe for pipe in all_pipelines if pipe in best_pipeline]))}
for i, _shots in enumerate(shot_numbers):
    for j, _lambda in enumerate(noise_rates):
        best_pipeline_id[i, j] = pipeline_ids[best_pipeline[i,j].item()]


# -

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
# %matplotlib notebook
plot_all_pipelines = False
if plot_all_pipelines:
    figsize = (9, 3+3*len(pipelines))
    fig, ax = plt.subplots(len(pipelines)+1, 1, figsize=figsize)
else:
    figsize = (10, 5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = [ax]

titles = {k:k for k in pipelines.keys()}
#     'None': "No Postprocessing",
#     'sdp': "Best Regularization", # SDP is best of the three regularization methods alone.
#     'displace': "Displacing", # Worse average than SDP alone
#     'thresh': "Thresholding", # Worse worst output than SDP alone
#     'single': "Single Rate Mitigation (Single)",
#     'avg': "Single Rate Mitigation (Average)",
#     'avg_sdp': "Single Rate Mitigation (Average) and Regularization",
#     'displace_avg_sdp': "Displacing, Single Rate Mitigation (Average) and Regularization",
#     'displace_avg': "Displacing and Single Rate Mitigation (Average)",
#     'split': "Feature-dependent Rate Mitigation",
#     'split_sdp': "Feature-dependent Rate Mitigation and Regularization",
#     'displace_split_sdp': "Displacing, Feature-dependent Rate Mitigation and Regularization",
# }
shot_coords = {_shots: shot_numbers.index(_shots)+0.5 for _shots in shot_numbers}
noise_coords = {_lambda: _lambda / (noise_rates[1]-noise_rates[0]) + 0.5 for _lambda in noise_rates}

# print(f"Worst performances:")
if plot_all_pipelines:
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
    #                 linecolor='k',
    #                 xticklabels=list(df['base_noise_rate'].unique()[::5]),
    #                 xticks=list(df['base_noise_rate'].unique()[::5]),
                    yticklabels=list(np.round(df['shots_sort'].astype(int).unique()[:-1],0))+['analytic'],
                   )
        ax[i].set_xticks([])
        ax[i].set_xlabel('')
        plt.setp( ax[i].yaxis.get_majorticklabels(), rotation=0 )
        ax[i].set_ylabel('# Measurements')
        ax[i].set_title(titles[pipeline_name])

        cbar = ax[i].collections[0].colorbar
        # Tick 1
        tick_col = 'k' if min_alignment > 0.2 else '1'
        cbar.ax.hlines(min_alignment, -1.2, 1.2, color=tick_col)
        cbar.ax.text(-1.5, min_alignment, f"{min_alignment:.2f}", horizontalalignment='right', verticalalignment='center')
        ax[i].plot(noise_coords[min_finite_df['base_noise_rate'].item()], shot_coords[min_finite_df['shots_sort'].item()], marker='x', color=tick_col)
        if min_df['shots'].item() == 0:
            # Tick 2 in case tick 1 was in the analytic domain
            tick_col = 'k' if min_alignment_finite > 0.2 else '1'
            cbar.ax.hlines(min_alignment_finite, -1.2, 1.2, color=tick_col)
            cbar.ax.text(-1.5, min_alignment_finite, f"{min_alignment_finite:.2f}", horizontalalignment='right', verticalalignment='center')
            ax[i].plot(noise_coords[min_finite_df['base_noise_rate'].item()],
                       shot_coords[min_finite_df['shots_sort'].item()], marker='x', color=tick_col)
    #     print(np.min(subdf['alignment']), np.max(subdf['alignment']))

#         print(f"{titles[pipeline_name]} - {min_alignment}")
    

i = len(ax)-1

best_df_pivot = best_df.pivot('shots_sort', 'base_noise_rate', 'alignment')
plot = sns.heatmap(data=best_df_pivot,
            vmin=-1,
            vmax=1+1e-5,
            cbar=True,
            ax=ax[i],
#             linewidth=0.5,
#             annot=best_pipeline_id,
            yticklabels=list(np.round(df['shots_sort'].astype(int).unique()[:-1],0))+['analytic'],
           )
ax[i].set_xticklabels([ticklabel.get_text()[:4] for ticklabel in ax[i].get_xticklabels()])
ax[i].set_xticks(ax[i].get_xticks()[::5])
plt.setp( ax[i].xaxis.get_majorticklabels(), rotation=0 )
ax[i].set_xlabel('Base noise rate $\\lambda_0$')
ax[i].set_ylabel('# Measurements')
# ax[i].set_title('Best')

for i1, _shots in enumerate(shot_numbers):
    for j1, _lambda in enumerate(noise_rates):
        if j1<len(noise_rates)-1 and vert[i1,j1]:
            _x = [noise_coords[_lambda]+0.5, noise_coords[_lambda]+0.5]
            _y = [shot_coords[_shots]-0.5, shot_coords[_shots]+0.5]
            ax[i].plot(_x, _y, color='k', linewidth=0.3)
        if i1<len(shot_numbers)-1 and horz[i1,j1]:
            _x = [noise_coords[_lambda]-0.5, noise_coords[_lambda]+0.5]
            _y = [shot_coords[_shots]+0.5, shot_coords[_shots]+0.5]
            ax[i].plot(_x, _y, color='k', linewidth=1., zorder=100)
cells = get_cells(vert, horz, 50)
centers = get_cell_label_pos(cells)
revert_pipeline_ids = {v :k for k,v in pipeline_ids.items()}
legend_entries = []
handles = []
for _id, coord in centers.items():
    indx = (np.round(coord).astype(int))
    
    pipe_id = best_pipeline_id[indx[0],indx[1]]
    ax[i].text(coord[1]+0.5, coord[0]+0.5, str(pipe_id), ha='center', va='center', bbox=dict(boxstyle="circle",
                   ec=(0., 0., 0.),
                   fc=(1., 1., 1.),
                alpha=0.1,  
                   ),
              alpha=0.8)
    if int(pipe_id) not in handles:
        legend_entries.append((int(pipe_id), revert_pipeline_ids[int(pipe_id)]))
        handles.append(int(pipe_id))
plt.setp( ax[i].yaxis.get_majorticklabels(), rotation=0 )
legend_entries = sorted(list(set(legend_entries)), key=lambda x: x[0])
handles = [AnyObject(han) for han, lab in legend_entries]
ax[i].legend(
    handles, 
    [(lab if lab!='' else 'No postprocessing') for han, lab in legend_entries],    
    handler_map={han:AnyObjectHandler() for han in handles},
    ncol=4,
    bbox_to_anchor=[0,1.],
    loc='lower left',
    labelspacing=0.8,
)
    
plt.tight_layout()
plt.savefig('mitigation_plots/best_postprocessing_Checkerboard_trained.pdf')


# +
# %matplotlib notebook
plot_pipelines = ['', 'thresh']

figsize = (9, 2.5*len(plot_pipelines))
fig, ax = plt.subplots(len(plot_pipelines), 1, figsize=figsize)


titles = {k:k for k in pipelines.keys()}
#     'None': "No Postprocessing",
#     'sdp': "Best Regularization", # SDP is best of the three regularization methods alone.
#     'displace': "Displacing", # Worse average than SDP alone
#     'thresh': "Thresholding", # Worse worst output than SDP alone
#     'single': "Single Rate Mitigation (Single)",
#     'avg': "Single Rate Mitigation (Average)",
#     'avg_sdp': "Single Rate Mitigation (Average) and Regularization",
#     'displace_avg_sdp': "Displacing, Single Rate Mitigation (Average) and Regularization",
#     'displace_avg': "Displacing and Single Rate Mitigation (Average)",
#     'split': "Feature-dependent Rate Mitigation",
#     'split_sdp': "Feature-dependent Rate Mitigation and Regularization",
#     'displace_split_sdp': "Displacing, Feature-dependent Rate Mitigation and Regularization",
# }
shot_coords = {_shots: shot_numbers.index(_shots)+0.5 for _shots in shot_numbers}
noise_coords = {_lambda: _lambda / (noise_rates[1]-noise_rates[0]) + 0.5 for _lambda in noise_rates}

# print(f"Worst performances:")
for i, pipeline_name in enumerate(plot_pipelines):
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
#                 linecolor='k',
#                 xticklabels=list(df['base_noise_rate'].unique()[::5]),
#                 xticks=list(df['base_noise_rate'].unique()[::5]),
                yticklabels=list(np.round(df['shots_sort'].astype(int).unique()[:-1],0))+['analytic'],
               )
    ax[i].set_xticks([])
    ax[i].set_xlabel('')
    plt.setp( ax[i].yaxis.get_majorticklabels(), rotation=0 )
    ax[i].set_ylabel('# Measurements')
#     ax[i].set_title(titles[pipeline_name])

    cbar = ax[i].collections[0].colorbar
    # Tick 1
    tick_col = 'k' if min_alignment > 0.2 else '1'
    cbar.ax.hlines(min_alignment, -1.2, 1.2, color=tick_col)
    cbar.ax.text(-1.5, min_alignment, f"{min_alignment:.2f}", horizontalalignment='right', verticalalignment='center')
    ax[i].plot(noise_coords[min_finite_df['base_noise_rate'].item()], shot_coords[min_finite_df['shots_sort'].item()], marker='x', color=tick_col)
    if min_df['shots'].item() == 0:
        # Tick 2 in case tick 1 was in the analytic domain
        tick_col = 'k' if min_alignment_finite > 0.2 else '1'
        cbar.ax.hlines(min_alignment_finite, -1.2, 1.2, color=tick_col)
        cbar.ax.text(-1.5, min_alignment_finite, f"{min_alignment_finite:.2f}", horizontalalignment='right', verticalalignment='center')
        ax[i].plot(noise_coords[min_finite_df['base_noise_rate'].item()],
                   shot_coords[min_finite_df['shots_sort'].item()], marker='x', color=tick_col)
#     print(np.min(subdf['alignment']), np.max(subdf['alignment']))

#         print(f"{titles[pipeline_name]} - {min_alignment}")


    
plt.tight_layout()
plt.savefig('mitigation_plots/improvement_postprocessing_Checkerboard_trained.pdf')
# -

best_df.loc[(df['base_noise_rate']==0.088) & (df['shots_sort']==int(1e10))]


