# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
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
# ### This notebook takes approximately ?? (??) minutes on a laptop without (with) recomputing the post-processing

# + tags=[]
import time
import warnings
from itertools import product
from functools import partial
from dill import load
import numpy as onp

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
# -

# ### Configuration

# +
# Deactivate the following flag to recompute the mitigated matrices instead of loading them from repository.
reuse_mitigated_matrices = True
# Make sure this matches the setting in the data generation notebook
use_trained_params = False
num_wires = 5
# If activated (default), this skips post-processing pipelines that are redundant/unreasonable
filter_pipelines = True
filename = f'data/noisy_sim/kernel_matrices_Checkerboard_{"" if use_trained_params else "un"}trained.zip'

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
# -

# ### Load target vector with training labels

np.random.seed(43)
_, y_train, _, _ = checkerboard(30, 30, 4, 4)

# ### Load raw kernel matrices

# +
# kernel_matrices = load(open(filename, 'rb+'))
# print(len(kernel_matrices))
# print(f"Noise_probabilities: {sorted(set(np.round(k[0], 3) for k in kernel_matrices.keys()))}")
# print(f"Shots: {sorted(set(k[1] for k in kernel_matrices.keys()))}")

# +
# kernel_matrices[(0.01, 30, 4)].shape
# -

# ### Set up pipelines for postprocessing

# +
# Regularization methods.
r_Tikhonov = qml.kernels.displace_matrix
r_thresh = qml.kernels.threshold_matrix
# Warning: The results strongly depend on the SDP solver. The MOSEK solver performs significantly
# better than CVXOPT.
r_SDP = partial(qml.kernels.closest_psd_matrix, fix_diagonal=True, solver="MOSEK")

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
if filter_pipelines:
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
        


def pipeline_sorting_key(x):
    """A key for sorting postprocessing pipelines.
    
    First, the number of applied functions in the pipeline (between 1 and 3
    for all pipelines except "No post-processing") is considered, then
    whether they start with a regularization (first) or mitigation (later).
    """
    reg_or_mit = {
        'No post-processing': -1,
        **{function_names[fun]: 0 for fun in regularizations},
        **{function_names[fun]: 1 for fun in mitigations},
    }
    substrings = x.split(', ')
    return (len(substrings), reg_or_mit[substrings[0]])


# -

# ### Load postprocessed matrices or apply postprocessing pipelines

# +
# Do not manually alter the following flag, unless you know what you are doing.
mitigated_filename = filename.split(".")[0] + '_mitigated.zip'
actually_reuse_mitigated_matrices = reuse_mitigated_matrices
if actually_reuse_mitigated_matrices:
    try:
        df = pd.read_pickle(mitigated_filename)
    except FileNotFoundError as e:
        print(e)
        # Can not reuse mitigated matrices if the file is not found...
        actually_reuse_mitigated_matrices = False
        
if not actually_reuse_mitigated_matrices:
    columns = ['base_noise_rate', 'shots', 'repetition', 'pipeline', 'alignment', 'target_alignment', 'q', 'shots_sort']
    exact_matrix = kernel_matrices[(0., 0)]
    target = np.outer(y_train, y_train)
    times_per_fun = {fun :0 for fun in regularizations+mitigations}
    fun_evals = {fun: 0 for fun in regularizations+mitigations}
    data = []
    no_pipeline_align = {
        key: float(qml.math.frobenius_inner_product(mat, exact_matrix, normalize=True))
        for key, mat in kernel_matrices.items()
    }

    for pipeline_name, pipeline in tqdm(filtered_pipelines.items(), total=len(filtered_pipelines)):
        print(pipeline_name)
        for key, mat in tqdm(kernel_matrices.items(), total=len(kernel_matrices)):
            K = np.copy(mat)
            if len(K.shape)>2:
                K = K[0]
            
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
                    target_align = np.nan
                    if fun==r_SDP:
                        print(e)
                    break
            else:
                normK = np.linalg.norm(K, 'fro')
                if np.isclose(normK, 0.):
                    align = np.nan
                    target_align = np.nan
                else:
                    align = float(qml.math.frobenius_inner_product(K, exact_matrix, normalize=True))
                    target_align = float(qml.math.frobenius_inner_product(K, target, normalize=True))
            no_pipe_align = no_pipeline_align[key]
            data.append([
                np.round(key[0], 3),
                key[1],
                key[2] if len(key)==3 else 0,
                pipeline_name,
                align,
                target_align,
                (align - no_pipe_align) / (1. - no_pipe_align) if no_pipe_align!=1. else 0.,
                key[1] if key[1]>0 else int(1e10), # We will use shots=1e10 for the analytic case to obtain correctly sorted results
            ])
    df = pd.DataFrame(data, columns=columns)
            
try:
    print(f"Average execution times of postprocessing functions:")
    for key, num in fun_evals.items():
        if num>0:
            print(f"{key}  -  {times_per_fun[key]/num}")
except:
    pass
# -


# ### Store postprocessed matrices

df.reset_index(level=0, inplace=True, drop=True)
df.to_pickle(mitigated_filename)

# Helper functions to pick parts of a dataframe with desired specs
same_shots = lambda dataframe, shots_sort: dataframe.loc[dataframe.shots_sort==shots_sort]
same_noise = lambda dataframe, base_noise_rate, shots_sort: dataframe.loc[(dataframe.base_noise_rate==base_noise_rate)&(dataframe.shots_sort==shots_sort)]
same_pipeline = lambda dataframe, pipeline: dataframe.loc[dataframe.pipeline==pipeline]
best_by_rounded_q = lambda dataframe, num_decimals: dataframe.loc[dataframe.q.round(num_decimals).idxmax()]

# +
# Compute the average alignment improvement across all samples per noise setting and pipeline.
mean_df = df.groupby(["base_noise_rate", "shots_sort", "pipeline"])["q"].mean()

# Restructure the dataframe
for i in range(3):
    mean_df = mean_df.reset_index(level=0, drop=False, inplace=False)
df = mean_df

# +
all_shots = sorted(df['shots_sort'].unique())[::-1]
all_noise_rates = sorted(df['base_noise_rate'].unique())
num_shots = len(all_shots)
num_noise_rates = len(all_noise_rates)
delta_noise_rate = all_noise_rates[1]-all_noise_rates[0]

def obtain_best_pipelines(dataframe):
    all_pipelines = sorted(dataframe['pipeline'].unique(), key=pipeline_sorting_key)
    num_pipelines = len(all_pipelines)

    best_pipeline = onp.zeros((num_shots, num_noise_rates), dtype=object)
    best_pipeline_id = onp.zeros((num_shots, num_noise_rates), dtype=int)
    vert = np.zeros((num_shots, num_noise_rates-1), dtype=bool)
    horz = np.zeros((num_shots-1, num_noise_rates), dtype=bool)

    best_data = []
    cumulative_q = {pipe: 0. for pipe in all_pipelines}
    for i, shots in tqdm(enumerate(all_shots), total=num_shots):
        for j, bnr in enumerate(all_noise_rates):
            this_noise_df = same_noise(dataframe, bnr, shots)

            # This loop only is relevant to the cumulative_q score, which currently is not used anymore.
            for pipeline in all_pipelines:
                # If the pipeline was deleted from the cumulative score because it failed
                # for a different noise setting, ignore it here.
                if pipeline not in cumulative_q:
                    continue
                q = same_pipeline(this_noise_df, pipeline).q.item()

                if not np.isnan(q):
                    cumulative_q[pipeline] += q
                else:
                    # Remove the pipeline from the cumulative score
                    del cumulative_q[pipeline]
            this_noise_best_df = best_by_rounded_q(this_noise_df, num_decimals)

            # Store the Series belonging to the best pipeline to the global best_df
            best_data.append(this_noise_best_df.to_list())
            # Store the best pipeline name and whether to draw boundaries in the cell heatmap
            best_pipeline[i, j] = this_noise_best_df.pipeline
            if j>0 and best_pipeline[i,j-1]!=best_pipeline[i,j]:
                vert[i,j-1] = True
            if i>0 and best_pipeline[i-1,j]!=best_pipeline[i,j]:
                horz[i-1,j] = True
    best_df = pd.DataFrame(best_data, columns=["pipeline", "shots_sort", "base_noise_rate", "q"])

    # Create a map of ids, mapping a pipeline name to an integer id, for the pipelines that occur in best_pipeline.
    pipeline_ids = {pipe: i for i, pipe in enumerate([pipe for pipe in all_pipelines if pipe in best_pipeline])}
    # Apply this map to the best_pipeline array
    for i, shots in enumerate(all_shots):
        for j, bnr in enumerate(all_noise_rates):
            best_pipeline_id[i, j] = pipeline_ids[best_pipeline[i,j]]
            
    return pipeline_ids, best_pipeline, best_pipeline_id, vert, horz, best_df

unfilt_pipeline_ids, unfilt_best_pipeline, unfilt_best_pipeline_id, unfilt_vert, unfilt_horz, unfilt_best_df = obtain_best_pipelines(df)
# -

# ### Group pipelines into "performance and stability groups"

# +
"""Sort the pipelines into groups:
1. Never produces a negative q
2. For a given number of shots, produces exclusively positive q below some survival probability threshold,
   and exclusively negative q above this threshold.
3. For a given number of shots, produces exclusively negative q below some survival probability threshold,
   and exclusively positive q above this threshold.
4. Produces a mixture of positive and negative q.

A pipeline that is not applicable/fails is counted with q=-20.
"""
# The threshold that is considered to be "0" to avoid sensitivity to rounding errors
zero = -1e-3

# Copy the df and set missing data for q to -20
na_is_neg_df = df.copy()
na_is_neg_df.loc[na_is_neg_df.q.isna(), "q"] = -20

groups = {1: [], 2: [], 3: [], 4: []}
unfilt_all_pipelines = sorted(df['pipeline'].unique(), key=pipeline_sorting_key)

for pipeline in unfilt_all_pipelines:
    this_pipe_df = same_pipeline(na_is_neg_df, pipeline)

    # Filter out group 1
    if all(this_pipe_df.q>=zero):
        groups[1].append(pipeline)
        continue
        
    # Check in which group a pipeline belongs per number of shots.
    group_features = []
    for shots in all_shots:
        this_df = same_shots(this_pipe_df, shots)
        # Check whether all negative entries are in one group from the first base noise rate onwards
        neg_ids = np.where(this_df.q<zero)[0]
        neg_is_contiguous = np.allclose(neg_ids, list(range(len(neg_ids))))
        # Check whether all non-negative entries are in one group from the first base noise rate onwards
        nonneg_ids = np.where(this_df.q>=zero)[0]
        nonneg_is_contiguous = np.allclose(nonneg_ids, list(range(len(nonneg_ids))))
        
        # Sort into group, based on data for this shots number
        if nonneg_is_contiguous and len(nonneg_ids)>0:
            group_features.append(2)
        elif neg_is_contiguous:
            group_features.append(3)
        else:
            group_features.append(4)
            
    # Conclude the group for the pipeline based on all group features per number of shots
    # group 2 (3) only is returned if *all* features are 2 (3).
    if all(f==2 for f in group_features):
        groups[2].append(pipeline)
    elif all(f==3 for f in group_features):
        groups[3].append(pipeline)
    else:
        groups[4].append(pipeline)

# MANUALLY move r_Tikhonov from group 4 to group 3, 
# because there are only 2 exceptions that shift it into group 4.
del groups[4][groups[4].index("r_Tikhonov")]
groups[3].append("r_Tikhonov")
warnings.warn("The pipeline 'r_Tikhonov' has been manually shifted from group 4 to group 3!")

# Return some statistics:
for i, g in groups.items():
    group_bests = [p for p in g if p in unfilt_pipeline_ids]
    print(f"Group {i} contains {len(g)} pipelines. ({len(group_bests)} of them are best at some point).")
    
print(f"\nGroup 1 (no negatives): {groups[1]}\n")
print(f"\n             best ones: {[p for p in groups[1] if p in unfilt_pipeline_ids]}\n")
print(f"Group 2 (no negatives below a certain noise level): {groups[2]}\n")
print(f"\n                                       best ones: {[p for p in groups[2] if p in unfilt_pipeline_ids]}\n")
print(f"Group 3 (no negatives above a certain noise level): {groups[3]}\n")
print(f"\n                                       best ones: {[p for p in groups[3] if p in unfilt_pipeline_ids]}\n")
# print(f"Group 4 (the rest)                                : {groups[4]}\n")
# print(f"\n                                       best ones: {[p for p in groups[4] if p in unfilt_pipeline_ids]}\n")

accepted_pipelines = groups[1] + groups[2] + groups[3]

# Restrict the data to the accepted pipelines in groups 1 to 3
accepted_df = df.loc[df.pipeline.isin(accepted_pipelines)]

# Obtain the best results from the data restricted to the accepted pipelines
acc_pipeline_ids, acc_best_pipeline, acc_best_pipeline_id, acc_vert, acc_horz, acc_best_df = obtain_best_pipelines(accepted_df)


# -

def compare_performances(old_best_pipeline, new_best_pipeline, old_dataframe, new_dataframe):
    factors = {}
    for i, shots in enumerate(all_shots):
        for j, bnr in enumerate(all_noise_rates):
            if new_best_pipeline[i, j] != old_best_pipeline[i, j]:
                old_sub_df = same_pipeline(same_noise(old_dataframe, bnr, shots), old_best_pipeline[i, j])
                old_q = old_sub_df.q.item()
                old_pipeline = old_sub_df.pipeline.item()
                new_sub_df = same_pipeline(same_noise(new_dataframe, bnr, shots), new_best_pipeline[i, j])
                new_q = new_sub_df.q.item()
                new_pipeline = new_sub_df.pipeline.item()
                factors[(shots, bnr, old_pipeline, new_pipeline)] = new_q / old_q
    return factors


# ### Compare filtered to unfiltered best performances

# +
# %matplotlib notebook
orig_to_acc_factors = compare_performances(unfilt_best_pipeline, acc_best_pipeline, df, accepted_df)

# Let's take a look at how the performance compares between the full data set and the filtered one.
factors = onp.array(list(orig_to_acc_factors.values()))
fig, ax = plt.subplots(1, 1)
ax.hist(factors, bins=100);
ax.set(xlabel="New performance/Old performance", ylabel='frequency')
print(f"The best pipeline changed for {len(factors)} / {num_noise_rates * num_shots} noise configurations.")
print(f"The largest performance decrease is {(1-np.min(factors))*100:.2f}%")
print(f"The median performance decrease is {(1-np.median(factors))*100:.2f}%")
print(f"The average performance decrease is {(1-np.mean(factors))*100:.2f}%")
for threshold in [0.01, 0.05, 0.1]:
    print(f"There are {len(factors[factors<1-threshold])} entries with performance decrease >{int(threshold*100):d}%")
# -

# ### Alternative, manual filtering step

# +
"""Here we filter the data in a different way, by keeping only a few most successful pipelines as well as the 
do-nothing pipeline. This choice is somewhat arbitrary because it involves a cutoff as to how often 
pipeline needs to be the best in order to be kept in the following selection. However, there is a large
gap between those pipelines that perform best on a sizeable fraction of all experiments and those that
only perform best for a few settings.
"""

final_pipelines = [
    "No post-processing",
    "r_Tikhonov",
    "r_thresh",
    "m_mean, r_SDP",
    "m_split, r_Tikhonov",
    "m_single, r_Tikhonov",
]

# Restrict the data to the accepted pipelines in final_pipelines
final_df = df.copy()
final_df = final_df.loc[final_df.pipeline.isin(final_pipelines)]

# Obtain the best results from the data restricted to the accepted pipelines
fin_pipeline_ids, fin_best_pipeline, fin_best_pipeline_id, fin_vert, fin_horz, fin_best_df = obtain_best_pipelines(final_df)


# Again compare the performance before and after the second, manual filtering step
acc_to_final_factors = compare_performances(unfilt_best_pipeline, fin_best_pipeline, df, final_df)
warnings.warn("Comparing the manually filtered pipelines to the completely unfiltered, not to the auto-filtered pipelines")

# Let's take a look at how the performance compares between the full data set and the filtered one.
factors = onp.array(list(acc_to_final_factors.values()))
fig, ax = plt.subplots(1, 1)
ax.hist(factors, bins=100);
ax.set(xlabel="New performance/Old performance", ylabel='frequency')
print(f"The best pipeline changed for {len(factors)} / {num_noise_rates * num_shots} noise configurations.")
print(f"The largest performance decrease is {(1-np.min(factors))*100:.2f}%")
print(f"The median performance decrease is {(1-np.median(factors))*100:.2f}%")
print(f"The average performance decrease is {(1-np.mean(factors))*100:.2f}%")
for threshold in [0.01, 0.05, 0.1]:
    print(f"There are {len(factors[factors<1-threshold])} entries with performance decrease >{int(threshold*100):d}%")


# -

# ### Helper functions for plotting contiguous cells of pipeline choices, custom labels

# +
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

# Legend handler artist to allow for Text/numeric handles, adds a rounded box around text
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


# -

# ### Create pretty-print versions of a pipeline name

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
    if pipe=='No post-processing':
        return pipe
    return ', '.join([fun_names[name] for name in pipe.split(", ")])


# +
# Choose the data to use for plots: unfiltered, filtered by groups, or manually filtered
# The plot in Fig. 6 shows the data for "manually filtered" as explained in the main text.
use_data = "manually filtered"
# use_data = "unfiltered"

if use_data=="unfiltered":
    pipeline_ids, best_pipeline, best_pipeline_id, vert, horz, best_df = (
        unfilt_pipeline_ids, unfilt_best_pipeline, unfilt_best_pipeline_id, unfilt_vert, unfilt_horz, unfilt_best_df
    )
elif use_data=="filtered by groups":
    pipeline_ids, best_pipeline, best_pipeline_id, vert, horz, best_df = (
        acc_pipeline_ids, acc_best_pipeline, acc_best_pipeline_id, acc_vert, acc_horz, acc_best_df
    )
elif use_data=="manually filtered":
    pipeline_ids, best_pipeline, best_pipeline_id, vert, horz, best_df = (
        fin_pipeline_ids, fin_best_pipeline, fin_best_pipeline_id, fin_vert, fin_horz, fin_best_df
    )

# Get boundary lines to draw
boundaries = []
for i, shots in enumerate(all_shots):
    shots_coord = all_shots.index(shots)
    for j, bnr in enumerate(all_noise_rates):
        bnr_coord = bnr / delta_noise_rate
        if j<num_noise_rates-1 and vert[i, j]:
            _x = [bnr_coord + 1., bnr_coord + 1.]
            _y = [shots_coord, shots_coord + 1.]
            boundaries.append((_x, _y))
        if i<num_shots-1 and horz[i, j]:
            _x = [bnr_coord, bnr_coord + 1.]
            _y = [shots_coord + 1., shots_coord + 1.]
            boundaries.append((_x, _y))
        
# Get labels and coordinates and prepare legend entries
cells = get_cells(vert, horz, None)
centers = get_cell_label_pos(cells)
revert_pipeline_ids = {v: k for k, v in pipeline_ids.items()}
legend_entries = []
texts = []
text_coords = []
for _id, coord in centers.items():
    x, y = np.round(coord).astype(int)
    # obtain the best pipeline
    pipe = best_pipeline[x, y]
    pipe_id = best_pipeline_id[x, y]
    texts.append(str(pipe_id))
    text_coords.append((coord[1]+0.5, coord[0]+0.5))
    if (pipe_id, pipe) not in legend_entries:
        legend_entries.append((pipe_id, pipe))

legend_entries = sorted(legend_entries, key=lambda x: x[0])
handles = [AnyObject(han) for han, _ in legend_entries]
labels = [prettify_pipeline(lab) for _, lab in legend_entries]  
handler_map = {han:AnyObjectHandler() for han in handles}

# +
# %matplotlib notebook
# Parameters for boundary drawing
sep_col1 = 'k'
sep_col2 = '1.0'
sep_lw = 1.1
# Parameters for cell labels
cell_label_kwargs = {
    "alpha": 1.,
    "color": "k",
    "fontsize": 11,
    "bbox": {
        "boxstyle": "round, pad=0.08",
        "ec": "1",
        "fc": "1",
        "alpha": 0.7,
    },
}
# fontsizes
xlabel_fs = 13.5
ylabel_fs = xlabel_fs
cbar_label_fs = xlabel_fs
tick_fs = 13.5
cbar_tick_fs = tick_fs
legend_handle_fs = 12
legend_label_fs = 10.5

figsize = (13, 7)

fig, ax = plt.subplots(1, 1, figsize=figsize)

best_df_pivot = best_df.pivot('shots_sort', 'base_noise_rate', 'q')
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

for x, y in boundaries:
    ax.plot(x, y, color=sep_col1, linewidth=sep_lw, zorder=98)

for pipe_id, (x, y) in zip(texts, text_coords):
    ax.text(x, y, pipe_id, ha='center', va='center', **cell_label_kwargs)

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
    ncol=3,
    bbox_to_anchor=(0.0, 1.),
    loc='lower left',
    labelspacing=0.75,
    borderpad=0.7,
    handletextpad=0.1,
    fontsize=legend_label_fs,
)

plt.tight_layout()
plt.savefig(plot_all_best_filename, bbox_inches='tight')

min_improve = best_df['q'].min()
max_improve_ana = best_df.loc[best_df.shots_sort>10000]['q'].max()
max_improve_nonana = best_df.loc[best_df.shots_sort<10000]['q'].max()
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
# This is an old plotting cell for a figure in the first version of the paper.

# xlabel_fs = 15
# ylabel_fs = xlabel_fs
# cbar_label_fs = xlabel_fs
# tick_fs = 15
# cbar_tick_fs = tick_fs

# max_tick_c = 'k'

# figsize = (6.5, 3.5)
# fig, ax = plt.subplots(1, 1, figsize=figsize)

# shot_coords = {_shots: all_shots.index(_shots)+0.5 for _shots in all_shots}
# noise_coords = {_lambda: _lambda / delta_noise_rate + 0.5 for _lambda in all_noise_rates}


# subdf = df.copy()
# subdf = subdf.loc[subdf['pipeline']==best_rated]
# print(subdf)
# #subdf.loc[:, 'relative'] = subdf.apply(relative_alignment, axis=1)
# subdf_pivot = subdf.pivot('shots_sort', 'base_noise_rate', 'q').sort_index(axis='rows', ascending=False)
# max_alignment = np.max(subdf['q'])
# min_alignment = np.min(subdf['q'])
# max_df = subdf.loc[[subdf['q'].idxmax()]]
# plot = sns.heatmap(
#     data=subdf_pivot,
#     vmin=0-1e-5,
#     vmax=0.5+1e-5,
# #     vmin=min_alignment,
# #     vmax=max_alignment,
#     cbar=True,
#     ax=ax,
#     cmap=mpl.cm.viridis,
#     linewidth=0.1,
#     yticklabels=(['analytic']+[int(s) for s in all_shots[1:]]),
# )
# cbar = ax.collections[0].colorbar

# # Tick for max improvement
# min_improve = subdf['q'].min()
# max_improve = subdf['q'].max()
# print(f"The pipeline <{best_rated}> improved the alignment by minimally "
#       f"{np.round(min_improve*100, 1)}% and maximally {np.round(max_improve*100, 1)}%.")
# max_df = subdf.loc[[subdf['q'].idxmax()]]
# cbar.ax.hlines(max_improve, -1.2, 1.2, color=max_tick_c)
# ax.plot(noise_coords[max_df['base_noise_rate'].item()], 
#            shot_coords[max_df['shots_sort'].item()], marker='o', markersize=4, color=max_tick_c)

# # Tick settings
# plt.setp( ax.yaxis.get_majorticklabels(), rotation=0 )
# ax.set_xticklabels([ticklabel.get_text()[:4] for ticklabel in ax.get_xticklabels()])
# ax.set_xticks(ax.get_xticks()[::5])

# plt.setp( ax.xaxis.get_majorticklabels(), rotation=0 )
# ax.set_xlabel('$1-\\lambda_0$', fontsize=xlabel_fs)
# ax.set_ylabel('Measurements  $M$', fontsize=ylabel_fs)
# cbar.ax.set_ylabel('Relative improvement $q$', fontsize=cbar_label_fs)
# ax.tick_params(labelsize=tick_fs)
# cbar.ax.tick_params(labelsize=cbar_tick_fs)

# formatter.set_rcParams()
# plt.tight_layout()
# plt.savefig(plot_single_best_filename, bbox_inches='tight')
# -

# ### Accepted pipelines before manual filtering
#
# The following plots show the performance in more detail for 
# - the manually filtered pipelines shown in Fig. 6, 
# - R-THR, M-MEAN, R-SDP, and
# - M-SPLIT, R-SDP to allow for a comparison with our hardware results (see Appendix of the paper, in particular Fig. 9)

# + tags=[]
FS = dict(ticks=14, texts=15, legend=15, labels=14)
fig, axs = plt.subplots(3, 1, figsize=(8, 10), gridspec_kw={"hspace": 0})
pipelines = sorted(
    list(best_df.pipeline.unique()) + ["m_split, r_SDP", "r_thresh, m_mean, r_SDP"],
    key=pipeline_sorting_key,
)

show_shots = [10000000000, 3000, 100]
show_df = df.loc[
    (df.pipeline.isin(pipelines))
    & (df.base_noise_rate<=0.08)
]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors_and_styles = list(product(colors[:len(pipelines)//3+1], ["-", "--", ":"]))
show_df["ppipeline"] = show_df.apply(lambda x: prettify_pipeline(x.pipeline), axis=1)
j = 0
for shots in df.shots_sort.unique():
    if shots not in show_shots:
        continue
    ax = axs[j]
    data_df = same_shots(show_df, shots)
#     data_df = data_df.loc[data_df.q.unique().len()>1]
    for (color, style), p in zip(colors_and_styles, pipelines):
        _data_df = data_df.loc[data_df.pipeline==p]
        ax.plot(
            _data_df.base_noise_rate, 
            _data_df.q, 
            label=(_data_df.ppipeline.unique().item() if j==0 else ""), 
            color=color,
#             marker="x",
            ls=style,
        )
    ylim = ax.get_ylim()
    lower_y = -0.5 if shots<1e5 else 0.55
    new_ylim = (max(lower_y, ylim[0]), min(1, ylim[1]))
    ax.set(xlabel="$1-\lambda_0$", ylim=new_ylim, ylabel="$q$")
    if j==0:
        leg = ax.legend(
            ncol=2,
            bbox_to_anchor=(0.5, 1.),
            loc="lower center",
            fontsize=FS["legend"],
        )

    ax.set_yticks(ax.get_yticks()[1:-1])
    ax.tick_params(labelsize=FS["ticks"])
    ax.text(
        0.06,
        new_ylim[1] - 0.2*(new_ylim[1]-new_ylim[0]),
        f"{int(shots)} shots" if shots < 1e5 else "analytic",
        bbox=dict(boxstyle="round, pad=0.22", facecolor="1", alpha=0.4),
        fontsize=FS["texts"],
    )
    ax.text(
        0.0015,
        ylim[1] - 0.04*(ylim[1]-ylim[0]),
        "abc"[j],
        fontsize=FS["texts"],
#         bbox=dict(boxstyle="round, pad=0.22", facecolor="1", alpha=0.7),
    )
    ax.xaxis.label.set_size(FS["labels"])
    ax.yaxis.label.set_size(FS["labels"])
    ax.set_xlim(-0.001, 0.08)
    if j<2:
        ax.xaxis.set_ticks([])
    j += 1
plt.tight_layout()
plt.savefig("images/postprocessing_lineplot.pdf")

# +
# This cell produces a preliminary, messy overview plot of more methods.

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
show_shots = [10000000000, 3000, 100]
show_df = df.loc[
    (df.shots_sort.isin(show_shots))
#     & (df.pipeline.isin(accepted_pipelines))
    & (df.pipeline.isin(best_df.pipeline.unique()))
]
show_df["shots"] = show_df.shots_sort.replace(10000000000, "analytic")
sns.lineplot(data=show_df, x="base_noise_rate", y="q", hue="pipeline", style="shots", ax=ax)

leg = ax.get_legend()
leg.set_bbox_to_anchor((1., 1.))
# leg.set_title(f"{int(shots)} shots")
ylim = ax.get_ylim()
ax.set(xlabel="$1-\lambda_0$", ylim=(max(-1.5, ylim[0]), min(1, ylim[1])))
plt.tight_layout()

# + tags=[]
# Look at the post-processing quality for a single pipeline - not a final plot but a useful tool

# Choose the pipeline
pipeline = "m_mean, r_SDP"

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
data = df.loc[(df.pipeline==pipeline)]
data["hue_shots"] = data.apply(lambda x: str(int(x.shots_sort)), axis=1)
sns.lineplot(data=data, x="base_noise_rate", y="q", hue="hue_shots", ax=ax)
leg = ax.get_legend()
leg.set_bbox_to_anchor((1., 1.))
leg.set_title(pipeline+"\n\n       shots")
ylim = ax.get_ylim()
ax.set(xlabel="$1-\lambda_0$", )
plt.tight_layout()
# -



