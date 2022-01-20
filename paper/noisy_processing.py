# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Noisy circuit simulations on Checkerboard dataset: Data Processing
#
# Here we process and plot the data from the data generation notebook on this dataset.
# This results in the heatmap figures in the paper, one in the results section and one in the appendix.
# You can decide to recompute the mitigated matrices by activating the corresponding flag in cell 2.
#
# ### This notebook takes approximately two minutes

# + tags=[]
import time
from itertools import product
from functools import partial
from dill import load, dump

from tqdm.notebook import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import rsmf

import pennylane as qml
from pennylane import numpy as np
from pennylane_cirq import ops as cirq_ops

import src.kernel_helper_functions as khf
from src.datasets import checkerboard

formatter = rsmf.setup(r"\documentclass[twocolumn,superscriptaddress,nofootinbib]{revtex4-2}")

# +
# Deactivate the following flag to recompute the mitigated matrices instead of loading them from repository.
reuse_mitigated_matrices = True
# Make sure this matches the setting in the data generation notebook
use_trained_params = False
num_wires = 5
# If activated (default), this skips post-processing pipelines that are redundant/unreasonable
_filter_pipelines = True
filename = f'data/noisy_sim/kernel_matrices_Checkerboard_{"" if use_trained_params else "un"}trained.dill'

# Set the method for rating the post-processing pipelines across all noise settings
# For our data, the two methods practically yield the same conclusions
choose_best_by = "number of top placements"
# choose_best_by = "total score"

# Set the number of decimals to round to when deciding which method is best.
# Reducing this number will make the analysis less differentiated but will provide more stable
# recommendations for the best post-processing pipeline across neighbouring noise settings
num_decimals = 5

plot_single_best_filename = f'images/globally_best_postprocessing_Checkerboard_{"" if use_trained_params else "un"}trained.pdf'
plot_all_best_filename = f'images/locally_best_postprocessing_Checkerboard_{"" if use_trained_params else "un"}trained.pdf'


# +
# Helper functions for cell plotting
def get_cells(vert, horz, iterations=None):
    """Given boolean boundary matrices, obtain cells via spreading iterations
    Args:
        vert (ndarray<bool>, shape=(m,n-1)): Vertical boundaries
        horz (ndarray<bool>, shape=(m-1,n)): Horizontal boundaries
        iterations=None (int): Number of spreading iterations. If None, defaults to max(m, n)
    
    Returns:
        mat (ndarray<int>, shape=(m,n)): Cells, given as labeling of matrix elements. The labels are contiguous.
    """
    num_rows = vert.shape[0] # This is m in the docstring.
    num_cols = horz.shape[1] # This is n in the docstring.
    if iterations is None:
        iterations = max(num_rows, num_cols)
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
                for _i, _j in nghbhood:
                    mat[_i,_j] = nghb_min
                    
    _map = {val: count for count, val in enumerate(np.unique(mat))}
    for i in range(num_rows):
        for j in range(num_cols):
            mat[i,j] = _map[mat[i,j]]
            
    return mat

def get_cell_centers(cells):
    """Get average coordinates per cell
    Args:
        cells (ndarray<int>): Cells as computed by `get_cells`
    Returns:
        centers (dict): Map from cell labels to mean coordinates of cells
        
    """
    centers = {}
    for _id in np.unique(cells):
        wheres = np.where(cells==_id)
        centers[_id] = np.array([np.mean(where.astype(float)) for where in wheres])
    return centers

def get_cell_label_pos(cells):
    """Get proper label position per cell
    Args:
        cells (ndarray<int>): Cells as computed by `get_cells`
    Returns:
        label_pos (dict): Map from cell labels to label coordinates for cells
        
    """
    label_pos = get_cell_centers(cells)
    ids = label_pos.keys()
    for _id in ids:
        center = label_pos[_id]
        x, y = map(int, np.round(center))
        if cells[x, y]!= _id:
            where = np.where(cells==_id)
            dists = [(coord, np.linalg.norm(center-coord,2)) for coord in zip(where[0], where[1])]
            label_pos[_id] = min(dists, key=lambda x: x[1])[0]
        
    return label_pos


# -

# # Load raw kernel matrices

kernel_matrices = load(open(filename, 'rb+'))

# +
# kernel_matrices[(0.0,0)]
# -

# # Get target matrix from training labels of Checkerboard dataset

np.random.seed(43)
_, y_train, _, _ = checkerboard(30, 30, 4, 4)


# # Set up pipelines for postprocessing

# +
# Regularization methods.
r_Tikhonov = qml.kernels.displace_matrix
r_thresh = qml.kernels.threshold_matrix
# Warning: The results strongly depend on the SDP solver. The MOSEK solver performs significantly
# better than CVXOPT.
r_SDP = partial(qml.kernels.closest_psd_matrix, solver="MOSEK", fix_diagonal=True)

# Device noise mititgation methods.
m_single = partial(qml.kernels.mitigate_depolarizing_noise, num_wires=num_wires, method="single")
m_mean = partial(qml.kernels.mitigate_depolarizing_noise, num_wires=num_wires, method="average")
m_split = partial(qml.kernels.mitigate_depolarizing_noise, num_wires=num_wires, method="split_channel")

# The "do-nothing" post-processing method, i.e. the identity.
Id = lambda mat: mat

# Names for pipeline keys
function_names = {
    r_Tikhonov: 'r_Tikhonov',
    r_thresh: 'r_thresh',
    r_SDP: 'r_SDP',
    m_single: 'm_single',
    m_mean: 'm_mean',
    m_split: 'm_split',
    Id: 'Id',
}

# All combinations of the shape regularize - mitigate - regularize
regularizations = [Id, r_Tikhonov, r_thresh, r_SDP]
mitigations = [Id, m_single, m_mean, m_split]
pipelines = list(product(regularizations, mitigations, regularizations))

filtered_pipelines = {}
if _filter_pipelines:
    for pipeline in pipelines:
        key = ', '.join([function_names[function] for function in pipeline if function!=Id])
        if key=='': # Give the Id-Id-Id pipeline a proper name
            key = 'No post-processing'
            
        if key in filtered_pipelines: # Skip duplicated keys (the dict would be overwritten anyways)
            continue
        if pipeline[0]==r_SDP and (pipeline[1]!=Id or pipeline[2]!=Id): # Skip r_SDP - ~ID - ~Id
            continue
        if pipeline[1]==Id and pipeline[2] in [r_Tikhonov, r_thresh]: # Skip regularize - Id - r_Tikhonov/thresh
            continue
        filtered_pipelines[key] = pipeline
        
else:
    for pipeline in pipelines:
        key = ', '.join([function_names[function] for function in pipeline])
        filtered_pipelines[key] = pipeline
# -

# # Apply mitigation techniques

# +
# Do not manually alter the following flag, unless you know what you are doing.
actually_reuse_mitigated_matrices = reuse_mitigated_matrices
if actually_reuse_mitigated_matrices:
    try:
        df = pd.read_pickle(filename[:-5]+'_mitigated.dill')
    except FileNotFoundError:
        # Can not reuse mitigated matrices if the file is not found...
        actually_reuse_mitigated_matrices = False
        
if not actually_reuse_mitigated_matrices:
    df = pd.DataFrame()
    exact_matrix = kernel_matrices[(0., 0)]
    target = np.outer(y_train, y_train)

    times_per_fun = {fun :0 for fun in regularizations+mitigations}
    fun_evals = {fun: 0 for fun in regularizations+mitigations}

    for pipeline_name, pipeline in tqdm(filtered_pipelines.items(), total=len(filtered_pipelines)):
        for key, mat in kernel_matrices.items():
            K = np.copy(mat)
            for fun in pipeline:
                try:
                    fun_start = time.process_time()
                    K = fun(K)
                    times_per_fun[fun] += time.process_time()-fun_start
                    fun_evals[fun] += 1
                    if np.any(np.isinf(K)):
                        raise ValueError
                except Exception as e:
                    K = None
                    align = np.nan
                    break
            else:
                normK = np.linalg.norm(K, 'fro')
                if np.isclose(normK, 0.):
                    align = np.nan
                    target_align = np.nan
                else:
                    align = float(qml.math.frobenius_inner_product(K, exact_matrix, normalize=True))
                    target_align = float(qml.math.frobenius_inner_product(K, target, normalize=True))
            df = df.append(pd.Series(
                {
                        'base_noise_rate': np.round(key[0], 3),
                        'shots': key[1],
                        'pipeline': pipeline_name,
                        'alignment': align,
                        'target_alignment': target_align,
                        'shots_sort': key[1] if key[1]>0 else int(1e10),
                    }),
                ignore_index=True,
                )
# -


df.reset_index(level=0, inplace=True, drop=True)
df.to_pickle(filename[:-5]+'_mitigated.dill')

# +
no_pipeline_df = df.loc[df['pipeline']=='No post-processing']

def relative_alignment(x):
    no_pipe_A = no_pipeline_df.loc[
        (no_pipeline_df['shots']==x['shots'])
        &(no_pipeline_df['base_noise_rate']==x['base_noise_rate'])
    ]['alignment'].item()
    if np.isclose(x['alignment'], 1.) and no_pipe_A==1.:
        return 0.
#     print(no_pipe_A, x['alignment'])
    return (x['alignment']-no_pipe_A)/(1-no_pipe_A)

def relative_target_alignment(x):
    no_pipe_A = no_pipeline_df.loc[
        (no_pipeline_df['shots']==x['shots'])
        &(no_pipeline_df['base_noise_rate']==x['base_noise_rate'])
    ]['target_alignment'].item()
#     print(no_pipe_A)
    
    return (x['target_alignment']-no_pipe_A)/(1-no_pipe_A)


# +
# print(df.pipeline.unique())
# -

try:
    print(f"Average execution times of postprocessing functions:")
    for key, num in fun_evals.items():
        if num>0:
            print(f"{key}  -  {times_per_fun[key]/num}")
except:
    pass


# +
# Find best pipeline for each combination of shots and noise_rate

def pipeline_sorting_key(x):
    reg_or_mit = {
        'No post-processing': -1,
        **{function_names[fun]: 0 for fun in regularizations},
        **{function_names[fun]: 1 for fun in mitigations},
    }
    substrings = x.split(', ')
    return (len(substrings), reg_or_mit[substrings[0]])

shot_numbers = sorted(list(df['shots_sort'].unique()))[::-1]
noise_rates = sorted(list(df['base_noise_rate'].unique()))
all_pipelines = sorted(list(df['pipeline'].unique()), key=pipeline_sorting_key)

best_pipeline = np.zeros((len(shot_numbers), len(noise_rates)), dtype=object)
best_pipeline_id = np.zeros((len(shot_numbers), len(noise_rates)), dtype=int)
vert = np.zeros((len(shot_numbers), len(noise_rates)-1), dtype=bool)
horz = np.zeros((len(shot_numbers)-1, len(noise_rates)), dtype=bool)
best_pipeline_target = np.zeros((len(shot_numbers), len(noise_rates)), dtype=object)
best_pipeline_id_target = np.zeros((len(shot_numbers), len(noise_rates)), dtype=int)

best_df = pd.DataFrame()
best_df_target = pd.DataFrame()
total_relative_score = {pipe: 0. for pipe in all_pipelines}
for i, _shots in tqdm(enumerate(shot_numbers), total=len(shot_numbers)):
    for j, _lambda in enumerate(noise_rates):
        sub_df = df.loc[(df['shots_sort']==_shots)&(df['base_noise_rate']==_lambda)]
        sub_df['relative'] = sub_df.apply(relative_alignment, axis=1)
        sub_df['relative_target'] = sub_df.apply(relative_target_alignment, axis=1)
        for pipe in all_pipelines:
            val = sub_df[sub_df.pipeline==pipe]['relative'].item()
            if pipe not in total_relative_score:
                continue
            elif not np.isnan(val):
                total_relative_score[pipe] += val
            else:
                del total_relative_score[pipe]
        
        best_sub_df = sub_df.loc[sub_df['alignment'].round(num_decimals).idxmax()]
        best_df = best_df.append(best_sub_df, ignore_index=True)
        best_sub_df_target = sub_df.loc[sub_df['target_alignment'].round(num_decimals).idxmax()]
        best_df_target = best_df_target.append(best_sub_df_target, ignore_index=True)

        best_pipeline[i, j] = best_sub_df['pipeline']
        best_pipeline_target[i, j] = best_sub_df_target['pipeline']
        if j>0 and best_pipeline[i,j-1]!=best_pipeline[i,j]:
            vert[i,j-1] = True
        if i>0 and best_pipeline[i-1,j]!=best_pipeline[i,j]:
            horz[i-1,j] = True

pipeline_ids = {pipe: i for i, pipe in enumerate([pipe for pipe in all_pipelines if pipe in best_pipeline])}
pipeline_ids_target = {pipe: i for i, pipe in enumerate(sorted([pipe for pipe in all_pipelines if pipe in best_pipeline_target]))}
for i, _shots in enumerate(shot_numbers):
    for j, _lambda in enumerate(noise_rates):
        best_pipeline_id[i, j] = pipeline_ids[best_pipeline[i,j].item()]
        best_pipeline_id_target[i, j] = pipeline_ids_target[best_pipeline_target[i,j].item()]


# +
# Legend handler artist to allow for Text/numeric handles, adds a round box around text

class AnyObject(object):
    def __init__(self, num, color='k'):
        self.my_text = str(num)
        self.my_color = color

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpl.text.Text(x=0, y=0, text=orig_handle.my_text, color=orig_handle.my_color,
                                verticalalignment=u'baseline', 
                                horizontalalignment=u'left', multialignment=None, 
                                fontproperties=None, linespacing=None,
                                fontsize=legend_handle_fs,
                                bbox=dict(boxstyle="round, pad=0.2",ec=(0., 0., 0.),fc=(1., 1., 1.), alpha=0.2),       
                                rotation_mode=None)
        handlebox.add_artist(patch)
        return patch


# +
fun_reg = {
    'r_Tikhonov': 'TIK',
    'r_thresh': 'THR',
    'r_SDP': 'SDP',
}
fun_mit = {
    'm_single': 'SINGLE',
    'm_mean': 'MEAN',
    'm_split': 'SPLIT',
}
fun_names = {
    **{k: f'$\\mathsf{{R}}\\mathrm{{-}}\\mathsf{{{v}}}$' for k, v in fun_reg.items()},
    **{k: f'$\\mathsf{{M}}\\mathrm{{-}}\\mathsf{{{v}}}$' for k, v in fun_mit.items()},
}

def prettify_pipeline(pipe):
    return ', '.join([fun_names[name] for name in pipe.split(", ")])


# +
# Get boundary lines to draw
shot_coords = {_shots: shot_numbers.index(_shots)+0.5 for _shots in shot_numbers}
noise_coords = {_lambda: _lambda / (noise_rates[1]-noise_rates[0]) + 0.5 for _lambda in noise_rates}
boundaries = []
for i, _shots in enumerate(shot_numbers):
    for j, _lambda in enumerate(noise_rates):
        if j<len(noise_rates)-1 and vert[i,j]:
            _x = [noise_coords[_lambda]+0.5, noise_coords[_lambda]+0.5]
            _y = [shot_coords[_shots]-0.5, shot_coords[_shots]+0.5]
            boundaries.append((_x, _y))
        if i<len(shot_numbers)-1 and horz[i,j]:
            _x = [noise_coords[_lambda]-0.5, noise_coords[_lambda]+0.5]
            _y = [shot_coords[_shots]+0.5, shot_coords[_shots]+0.5]
            boundaries.append((_x, _y))
        
# Get labels and coordinates and prepare legend entries
cells = get_cells(vert, horz, None)
centers = get_cell_label_pos(cells)
revert_pipeline_ids = {v: k for k,v in pipeline_ids.items()}
legend_entries = []
handles = []
texts = []
text_coords = []
for _id, coord in centers.items():
    x, y = np.round(coord).astype(int)
    pipe_id = best_pipeline_id[x, y]
    texts.append(str(pipe_id))
    text_coords.append((coord[1]+0.5, coord[0]+0.5))
    if int(pipe_id) not in handles:
        legend_entries.append((int(pipe_id), revert_pipeline_ids[int(pipe_id)]))
        handles.append(int(pipe_id))
legend_entries = sorted(legend_entries, key=lambda x: x[0])
handles = [AnyObject(han) for han, _ in legend_entries]
labels = [(prettify_pipeline(lab) if lab!='No post-processing' else lab) for _, lab in legend_entries]  
handler_map = {han:AnyObjectHandler() for han in handles}

# +
# %matplotlib notebook
# Parameters for boundary drawing
sep_col1 = 'k'
sep_col2 = '1.0'
sep_lw = 1.1
# Parameters for cell labels
cell_label_ec = '1'
cell_label_fc = '1'
cell_label_falpha = 0.7
cell_label_tc = 'k'
cell_label_talpha = 1.
cell_label_pad = 0.08
# fontsizes
xlabel_fs = 13.5
ylabel_fs = xlabel_fs
cbar_label_fs = xlabel_fs
tick_fs = 13.5
cbar_tick_fs = tick_fs
cell_label_fs = 11
legend_handle_fs = 12
legend_label_fs = 10.5

figsize = (13, 7)

fig, ax = plt.subplots(1, 1, figsize=figsize)

best_df_pivot = best_df.pivot('shots_sort', 'base_noise_rate', 'relative')
best_df_pivot = best_df_pivot.sort_index(axis='rows', ascending=False)
ticklabel_kwarg_to_heatmap = {
    'yticklabels': (list(np.round(df['shots_sort'].astype(int).unique()[:-1],0))+['analytic'])[::-1]
}
plot = sns.heatmap(data=best_df_pivot,
            vmin=0-1e-5,
            vmax=1+1e-5,
            cbar=True,
            ax=ax,
            cmap=mpl.cm.viridis,
            cbar_kws={'pad': 0.01},
            **ticklabel_kwarg_to_heatmap,
           )

for _x, _y in boundaries:
    ax.plot(_x, _y, color=sep_col1, linewidth=sep_lw, zorder=98)

for pipe_id, (x, y) in zip(texts, text_coords):
    ax.text(
        x,
        y,
        pipe_id,
        ha='center',
        va='center',
        bbox={
            'boxstyle': f"round, pad={cell_label_pad}",
            'ec': cell_label_ec,
            'fc': cell_label_fc,
            'alpha': cell_label_falpha,
        },
        alpha=cell_label_talpha,
        color=cell_label_tc,
        fontsize=cell_label_fs,
    )

cbar = ax.collections[0].colorbar

ax.set_xticklabels([ticklabel.get_text()[:4] for ticklabel in ax.get_xticklabels()])
ax.set_xticks(ax.get_xticks()[::5])
for axis in (ax.xaxis, ax.yaxis):
    plt.setp(axis.get_majorticklabels(), rotation=0)
ax.set_xlabel('$1-\\lambda_0$', fontsize=xlabel_fs)
ax.set_ylabel('Measurements  $M$', fontsize=ylabel_fs)
cbar.ax.set_ylabel('Relative improvement $q$', fontsize=cbar_label_fs)

ax.tick_params(labelsize=tick_fs)
cbar.ax.tick_params(labelsize=cbar_tick_fs)
# formatter.set_rcParams()
ax.legend(
    handles, 
    labels,
    handler_map=handler_map,
    ncol=4,
    bbox_to_anchor=(0.0, 1.),
    loc='lower left',
    labelspacing=0.75,
    borderpad=0.7,
    handletextpad=0.1,
    fontsize=legend_label_fs,
)

plt.tight_layout()
plt.savefig(plot_all_best_filename, bbox_inches='tight')

min_improve = best_df['relative'].min()
max_improve_ana = best_df.loc[best_df.shots==0]['relative'].max()
max_improve_nonana = best_df.loc[best_df.shots!=0]['relative'].max()
print(f"The locally best pipeline improved the alignment by minimally "
      f"{np.round(min_improve*100, 1)}% and maximally {np.round(max_improve_nonana*100, 1)}% (without M='analytic') " 
      f"or {np.round(max_improve_ana*100, 1)}% (for M='analytic').")
# -
# We globally rate the post-processing strategy as best which is the best for most noise parameters. 
if choose_best_by=="number of top placements":
    rating = {pipeline: np.sum(best_pipeline==pipeline).item() for pipeline in filtered_pipelines}
    best_rated = max(rating.items(), key=lambda x: x[1])[0]
    # Show the ratings for those pipelines that are best at least once:
    print(*sorted([rating for rating in rating.items() if rating[1]>0], key=lambda x: x[1], reverse=True), sep='\n')
elif choose_best_by=="total score":
#     print(total_relative_score)
    sorted_rating = sorted(total_relative_score.items(), key=lambda x: x[1], reverse=True)
    best_rated = sorted_rating[0][0]
    # Show the ratings
    print(*sorted_rating, sep='\n')

# +
xlabel_fs = 15
ylabel_fs = xlabel_fs
cbar_label_fs = xlabel_fs
tick_fs = 15
cbar_tick_fs = tick_fs

max_tick_c = 'k'

figsize = (6.5, 3.5)
fig, ax = plt.subplots(1, 1, figsize=figsize)

shot_coords = {_shots: shot_numbers.index(_shots)+0.5 for _shots in shot_numbers}
noise_coords = {_lambda: _lambda / (noise_rates[1]-noise_rates[0]) + 0.5 for _lambda in noise_rates}


subdf = df.loc[df['pipeline']==best_rated]
subdf.loc[:, 'relative'] = subdf.apply(relative_alignment, axis=1)
subdf_pivot = subdf.pivot('shots_sort', 'base_noise_rate', 'relative').sort_index(axis='rows', ascending=False)
max_alignment = np.max(subdf['relative'])
min_alignment = np.min(subdf['relative'])
max_df = subdf.loc[[subdf['relative'].idxmax()]]
plot = sns.heatmap(
    data=subdf_pivot,
    vmin=0-1e-5,
    vmax=0.5+1e-5,
#     vmin=min_alignment,
#     vmax=max_alignment,
    cbar=True,
    ax=ax,
    cmap=mpl.cm.viridis,
    linewidth=0.1,
    yticklabels=(list(np.round(df['shots_sort'].astype(int).unique()[:-1],0))+['analytic'])[::-1],
)
cbar = ax.collections[0].colorbar

# Tick for max improvement
min_improve = subdf['relative'].min()
max_improve = subdf['relative'].max()
print(f"The pipeline <{best_rated}> improved the alignment by minimally "
      f"{np.round(min_improve*100, 1)}% and maximally {np.round(max_improve*100, 1)}%.")
max_df = subdf.loc[[subdf['relative'].idxmax()]]
cbar.ax.hlines(max_improve, -1.2, 1.2, color=max_tick_c)
ax.plot(noise_coords[max_df['base_noise_rate'].item()], 
           shot_coords[max_df['shots_sort'].item()], marker='o', markersize=4, color=max_tick_c)

# Tick settings
plt.setp( ax.yaxis.get_majorticklabels(), rotation=0 )
ax.set_xticklabels([ticklabel.get_text()[:4] for ticklabel in ax.get_xticklabels()])
ax.set_xticks(ax.get_xticks()[::5])

plt.setp( ax.xaxis.get_majorticklabels(), rotation=0 )
ax.set_xlabel('$1-\\lambda_0$', fontsize=xlabel_fs)
ax.set_ylabel('Measurements  $M$', fontsize=ylabel_fs)
cbar.ax.set_ylabel('Relative improvement $q$', fontsize=cbar_label_fs)
ax.tick_params(labelsize=tick_fs)
cbar.ax.tick_params(labelsize=cbar_tick_fs)

formatter.set_rcParams()
plt.tight_layout()
plt.savefig(plot_single_best_filename, bbox_inches='tight')
