import seaborn as sns
import pennylane as qml
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools
import tqdm
import os
import sys
import numpy as np
import rsmf
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import scipy as sp

# +
# Random seed for resampling
np.random.seed(42)

# Filenames
filename_mitigated_matrices = 'data/mitigated_hardware_matrices.pkl'
filename_noiseless_matrix = 'data/noiseless_matrix_for_hardware.npy'
filename_prefix_QPU = 'data/QPU/ionq_kernel_matrix_'
filename_alignment_plot = 'images/hardware_alignment.pdf'

# If the following flag is set to True, the post-processing methods are recomputed otherwise the stored data will
# be used and plotted below. The recomputation will take about 15 minutes on a laptop computer. Note that the
# resampling seed was fixed above such that recomputing will not produce new numerics.
recompute_mitigation = False

# If the following flag is set to True, only the non-redundant an meaningful combinations of post-processing methods
# will be computed. Otherwise, 
_filter = True
# -

noiseless_kernel_matrix = np.load(filename_noiseless_matrix)


def resample_from_measurement_distribution(distribution, n_shots):
    """sample new shots from the sampled distribution of the QPU
    Args:
        distribution (dict): Probability distribution with signature str: float.
        n_shots (int): Number of samples to draw.
        
    Returns:
        kernel_entry (float): The frequency of the state '000' in the resampled distribution.
    
    Comments:
        If the values of distribution do not sum to 1, we distribute the error across all states.
    """
    states = list(distribution.keys())
    probabilities = np.array(list(distribution.values()))
    sum_probabilities = np.sum(probabilities)
    if sum_probabilities != 1:
        probabilities += (1-sum_probabilities)/len(probabilities)

    samples = np.random.choice(states, n_shots, p=probabilities)
    kernel_entry = np.mean(samples=='000')

    return kernel_entry


def translate_folder(module_path, n_shots):
    data = {'timestamp': [], 'measurement_result': []}
    for dirs, _, files in os.walk(module_path):
        for file in files:
            if file == 'results.json':
                with open(dirs + '/' + file) as myfile:
                    obj = json.loads(myfile.read())
                timestamp = obj['taskMetadata']['createdAt']
#                 print(timestamp)
                distribution = obj['measurementProbabilities']
                if n_shots==175:
                    try:
                        measurement_result = distribution['000']
                    except KeyError:
                        measurement_result = 0
                else:
                    measurement_result = resample_from_measurement_distribution(distribution, n_shots)
                time_for_sorting = timestamp[8:10] + \
                    timestamp[11:13] + timestamp[14:16] + timestamp[17:19]
                data['timestamp'].append(time_for_sorting)
                data['measurement_result'].append(measurement_result)
            else:
                print('not json')
    df = pd.DataFrame(data, columns=['timestamp', 'measurement_result'])
    df = df.sort_values(by=['timestamp'])
    return(df['measurement_result'])


df = pd.DataFrame()
n_shots_array = [15, 25, 50, 75, 100, 125, 150, 175]  # , 200, 500]
kernel_matrices = []
for n_shots in n_shots_array:
    kernel_array = np.zeros(1830)
    partition = [0, 679, 681, 929, 1229, 1529, 1530, 1830]
    slices = [(partition[i], partition[i+1]) for i in range(len(partition)-1)]
    for _slice in slices:
        data_path = os.path.abspath(os.path.join(f'{filename_prefix_QPU}{_slice[0]}_{_slice[1]}/'))
        kernel_array[slice(*_slice)] = translate_folder(data_path, n_shots)
    N_datapoints = 60
    kernel_matrix = np.zeros((N_datapoints, N_datapoints))
    index = 0
    for i in range(N_datapoints):
        for j in range(i, N_datapoints):
            kernel_matrix[i, j] = kernel_array[index]
            kernel_matrix[j, i] = kernel_matrix[i, j]
            index += 1

    alignment = qml.kernels.matrix_inner_product(kernel_matrix, noiseless_kernel_matrix, normalize=True)
    print(alignment, 'alignment')
    df = df.append({
        'n_shots': n_shots,
        'kernel_matrix': kernel_matrix,
        'alignment': alignment,
        'pipeline': 'No post-processing',
    }, ignore_index=True)

# +
# Pipeline definitions
num_wires = 3
r_Tikhonov = qml.kernels.displace_matrix
r_thresh = qml.kernels.threshold_matrix
r_SDP = qml.kernels.closest_psd_matrix

m_single = lambda mat: qml.kernels.mitigate_depolarizing_noise(mat, num_wires, method='single')
m_mean = lambda mat: qml.kernels.mitigate_depolarizing_noise(mat, num_wires, method='average')
m_split = lambda mat: qml.kernels.mitigate_depolarizing_noise(mat, num_wires, method='split_channel')

Id = lambda mat: mat

regularizations = [Id, r_Tikhonov, r_thresh, r_SDP]
mitigations = [Id, m_single, m_mean, m_split]
function_names = {
    r_Tikhonov: 'r_Tikhonov',
    r_thresh: 'r_thresh',
    r_SDP: 'r_SDP',
    m_single: 'm_single',
    m_mean: 'm_mean',
    m_split: 'm_split',
    Id: 'Id',
}
pipelines = list(itertools.product(regularizations, mitigations, regularizations))

def apply_pipeline(pipeline, mat):
    out = np.copy(mat)
    for function in pipeline:
        try:
            out = function(out)
            if np.any(np.isinf(out)):
                raise ValueError
        except Exception as e: # Catches problems in the SDP and problems caused by matrices that are too noisy for mitigation
            print(e)
            return None
        
    return out


# +
# Apply pipelines if activated/no file was found
actually_recompute_mitigation = False

if not recompute_mitigation:
    try:
        df = pd.read_pickle(filename_mitigated_matrices)
        print(len(df))
        actually_recompute_mitigation = False
    except FileNotFoundError:
        actually_recompute_mitigation = True

if actually_recompute_mitigation or recompute_mitigation:
    for n_shots in tqdm.notebook.tqdm(n_shots_array):
        noisy_kernel_matrix = df.loc[(df.n_shots==n_shots) & (df.pipeline=='No post-processing')].kernel_matrix.item()
        used_pipelines = set(['No post-processing'])
        for pipeline in pipelines:

            if _filter:
                key = ', '.join([function_names[function] for function in pipeline if function!=Id])
                if key=='': # Give the Id-Id-Id pipeline a proper name
                    key = 'No post-processing'
                if key in used_pipelines: # Skip duplicated keys (the dict would be overwritten anyways)
                    continue
                if pipeline[0]==r_SDP and (pipeline[1]!=Id or pipeline[2]!=Id): # Skip r_SDP - ~ID - ~Id
                    continue
                if pipeline[1]==Id and pipeline[2] in [r_Tikhonov, r_thresh]: # Skip regularize - Id - r_Tikhonov/thresh
                    continue
            else:
                key = ', '.join([function_names[function] for function in pipeline])
            mitigated_kernel_matrix = apply_pipeline(pipeline, noisy_kernel_matrix)
            if mitigated_kernel_matrix is None:
                alignment = None
            else:
                alignment = qml.kernels.matrix_inner_product(
                    mitigated_kernel_matrix, noiseless_kernel_matrix, normalize=True
                )
            df = df.append({
                'n_shots': n_shots,
                'kernel_matrix': mitigated_kernel_matrix,
                'alignment': alignment,
                'pipeline': key,
            }, ignore_index=True)
            used_pipelines.add(key)
# -

df.reset_index(level=0, inplace=True, drop=True)
df.to_pickle(filename_mitigated_matrices)

# +
noisy_df = df.loc[df.pipeline=='No post-processing']
mitigated_df = plot_df = df.loc[df.pipeline!='No post-processing']
num_top_pipelines = 2

def prettify_pipelines(x):
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
    funs = [fun_names[fun] for fun in x.pipeline.split(', ')]
    return ', '.join(funs)

def top_pipelines(n_shots, num_pipelines):    
    indices = mitigated_df.loc[mitigated_df.n_shots==n_shots].alignment.sort_values().index[-num_pipelines:]
    return mitigated_df.loc[indices]
def get_q(x):
    align = x.alignment
    raw_align = noisy_df.loc[noisy_df.n_shots==x.n_shots].alignment.item()
    return (align-raw_align)/(1-raw_align)

best_df = pd.DataFrame()
for n_shots in n_shots_array:
    best_df = pd.concat([best_df, top_pipelines(n_shots, 2)])
best_df['pretty_pipeline'] = best_df.apply(prettify_pipelines, axis=1)
best_df['q'] = best_df.apply(get_q ,axis=1)

# +
# %matplotlib notebook
formatter = rsmf.setup(r"\documentclass[twocolumn,superscriptaddress,nofootinbib]{revtex4-2}")
formatter.set_rcParams()
fig = formatter.figure(aspect_ratio=0.75, wide=False)
grid = mpl.gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
axs = [fig.add_subplot(grid[0,0])]
ms = 30
lw = 2
hue_order = list(best_df.pretty_pipeline.unique())
palette = sns.color_palette(n_colors=len(hue_order))

for i in range(2):
    marker = mpl.markers.MarkerStyle('o', fillstyle=['left','right'][i])
    df_i = best_df.loc[(best_df.pretty_pipeline==hue_order[i])]
    axs[0].scatter(df_i.n_shots, df_i.alignment, label=hue_order[i], color=palette[i],
                  marker=marker, s=ms, ec='1.', lw=0.2)

sns.scatterplot(data=noisy_df, x='n_shots', y='alignment', color='k', marker='d', label='No post-processing',
               s=ms, ax=axs[0],)
axs[0].set_xlabel(f"Measurements $M$")
axs[0].set_ylabel("Alignment A$(\overline{K}_M,K)$")
# axs[0].set_ylim((0.88, 1))
axs[0].set_xticks(n_shots_array)
handles, labels = axs[0].get_legend_handles_labels()

leg = axs[0].legend(handles[::-1], labels[::-1], loc='lower right')
plt.tight_layout()
plt.savefig(filename_alignment_plot)
# -

q_min = np.min(best_df.q)
q_max = np.max(best_df.q)
q_mean = np.mean(best_df.q)
print(f"The alignment improvement (q) was minimally  {q_min*100:.1f}%"
      f"\n{' '*34}maximally  {q_max*100:.1f}%"
      f"\n{' '*34}on average {q_mean*100:.1f}%"
       "\nfor the two best post-processing strategies.")


