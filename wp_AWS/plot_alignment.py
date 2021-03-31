import seaborn as sns
import pennylane as qml
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
formatter = rsmf.setup(r"\documentclass[twocolumn,superscriptaddress,nofootinbib]{revtex4-2}")

noiseless_kernel_matrix = np.load('noiseless_kernel_matrix.npy')


def sample_from_measurement_distribution(distribution, n_shots):
    values = list(distribution.keys())
    probabilities = np.array(list(distribution.values()))
    sum_probabilities = np.sum(probabilities)
    if sum_probabilities != 1:
        probabilities += (1-sum_probabilities)/len(probabilities)

    samples = np.random.choice(values, n_shots, p=probabilities)
    kernel_entry = np.sum(samples=='000')/n_shots

    return kernel_entry


def translate_folder(module_path, n_shots):
#     print(n_shots, 'shots')
    index = 1
    data_dict = {'timestamp': [], 'measurement_result': []}
    for dirs, subdir, files in os.walk(module_path):

        for file in files:
            if file == 'results.json':
                with open(dirs + '/' + file) as myfile:
                    data = myfile.read()
                obj = json.loads(data)
                timestamp = obj['taskMetadata']['createdAt']
                if n_shots == 175:
                    try:
                        measurement_result = obj['measurementProbabilities']['000']
                    except KeyError:
                        measurement_result = 0
                else:
                    measurement_result = sample_from_measurement_distribution(
                        obj['measurementProbabilities'], n_shots)
                time_for_sorting = timestamp[8:10] + \
                    timestamp[11:13] + timestamp[14:16] + timestamp[17:19]
                data_dict['timestamp'].append(time_for_sorting)
                data_dict['measurement_result'].append(measurement_result)
                index += 1
            else:
                print('not json')
    df = pd.DataFrame(data_dict, columns=['timestamp', 'measurement_result'])
    df = df.sort_values(by=['timestamp'])
    return(df['measurement_result'])


def visualize_kernel_matrices(kernel_matrices, draw_last_cbar=False):
    num_mat = len(kernel_matrices)
    width_ratios = [1]*num_mat+[0.2]*int(draw_last_cbar)
    fig, ax = plt.subplots(1, num_mat+draw_last_cbar, figsize=(num_mat *
                                                               5+draw_last_cbar, 5), gridspec_kw={'width_ratios': width_ratios})
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
            cbar=True
        )
    if draw_last_cbar:
        ch = plot.get_children()
        fig.colorbar(ch[0], ax=ax[-2], cax=ax[-1])
    plt.show()


if __name__ == "__main__":
    df = pd.DataFrame()
    kernel_array = []
    n_shots_array = [15, 25, 50, 75, 100, 125, 150, 175]  # , 200, 500]
    kernel_matrices = []
    for n_shots in n_shots_array:
        kernel_array = [0] * 1830
        
        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_0_679/'))
        kernel_array[:679] = translate_folder(module_path, n_shots)
        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_679_680/'))
        kernel_array[679:681] = translate_folder(module_path, n_shots)
        
        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_681_929'))
        kernel_array[681:681+248] = translate_folder(module_path, n_shots)
    
        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_929_1229'))
        kernel_array[929:1229] = translate_folder(module_path, n_shots)
        
        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_1229_1529'))
        kernel_array[1229:1529] = translate_folder(module_path, n_shots)

        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_1529'))
        kernel_array[1529] = translate_folder(module_path, n_shots)
        
        
        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_1529_1829'))
        kernel_array[1530:1830] = translate_folder(module_path, n_shots)
        
        #print(kernel_array[1829])
        N_datapoints = 60
        kernel_matrix = np.zeros((60, 60))
        index = 0
        for i in range(N_datapoints):
            for j in range(i, N_datapoints):
                kernel_matrix[i, j] = kernel_array[index]
                kernel_matrix[j, i] = kernel_matrix[i, j]
                index += 1
#         kernel_matrix = np.reshape(kernel_matrix, (60, 60))
        alignment = qml.kernels.matrix_inner_product(kernel_matrix, noiseless_kernel_matrix, normalize=True)
        print(alignment, 'alignment')
        df = df.append({
            'n_shots': n_shots,
            'kernel_matrix': kernel_matrix,
            'alignment': alignment,
            'pipeline': 'No post-processing',
        }, ignore_index=True)
     #   kernel_matrices.append(kernel_matrix)

    # visualize_kernel_matrices(kernel_matrices)

sns.pointplot(data=df, x='n_shots', y='alignment')

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
# Apply pipelines
recompute_pipelines = True
actually_recompute_pipelines = False
filter = False



if not recompute_pipelines:
    try:
        df = pd.read_pickle('mitigated_hardware_matrices.pkl')
        print(df)
        actually_recompute_pipelines = False
    except FileNotFoundError:
        actually_recompute_pipelines = True

if actually_recompute_pipelines or recompute_pipelines:
    for n_shots in tqdm.notebook.tqdm(n_shots_array):
        noisy_kernel_matrix = df.loc[(df.n_shots==n_shots) & (df.pipeline=='No post-processing')].kernel_matrix.item()
        used_pipelines = set(['No post-processing'])
        for pipeline in pipelines:

            if filter:
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

# +
noisy_df = df.loc[df.pipeline=='No post-processing']
mitigated_df = plot_df = df.loc[df.pipeline!='No post-processing']
num_top_pipelines = 1

def prettify_pipelines(x):
    funs = x.pipeline.split(', ')
    new_funs = []
    for fun in funs:
        if fun[0]=='m':
            new_fun = '$\\mathit{m}_\\mathrm{'
        else:
            new_fun = '$r_\mathrm{'
        new_fun += fun[2:]
        new_fun += '}$'
        new_funs.append(new_fun)
    return ', '.join(new_funs)

def fit_fun(M, a, b, c):
    return c-np.exp(-np.sqrt(M)*a+b)

def top_pipelines(n_shots):    
    indices = mitigated_df.loc[mitigated_df.n_shots==n_shots].alignment.sort_values().index[-num_top_pipelines:]
    return mitigated_df.loc[indices]

best_df = pd.DataFrame()
for n_shots in tqdm.notebook.tqdm(n_shots_array):
    best_df = pd.concat([best_df, top_pipelines(n_shots)])
best_df['pretty_pipeline'] = best_df.apply(prettify_pipelines, axis=1)

p_noisy, pcov_noisy = sp.optimize.curve_fit(fit_fun, n_shots_array, noisy_df.alignment.to_numpy(), p0=[1, 0, 1])
p_best, pcov_best = sp.optimize.curve_fit(fit_fun, n_shots_array, best_df.alignment.to_numpy(), p0=[1, 0, 1])

# +
# %matplotlib notebook
fs = 14
ms = 50
lw = 2

#print(best_df)
sns.scatterplot(data=best_df, x='n_shots', y='alignment', hue='pretty_pipeline',
                style='pretty_pipeline', markers=['o', 'X'], s=ms
                )

sns.scatterplot(data=noisy_df, x='n_shots', y='alignment', color='k', marker='d', label='No post-processing',
               s=ms)


plt.xlabel(f"# Measurements $M$", fontsize=fs)
plt.ylabel("Alignment $\\operatorname{A}(\overline{K},K)$", fontsize=fs)
plt.plot([10, 180], [noisy_df.alignment.max()]*2, ls='-', lw=lw/2, color='0.8', zorder=-10)
noisy_fit_label = "$"+str(np.round(p_noisy[2],2))+"-"+str(np.round(np.exp(p_noisy[1]),2))+"e^{-"+str(np.round(p_noisy[0],2))+"\\sqrt{M}}$"
plt.plot(range(10, 181), [fit_fun(n_shots, *p_noisy) for n_shots in range(10, 181)], ls=':', lw=lw,
         color='0.6', zorder=-10, label=noisy_fit_label)

best_fit_label = "$"+str(np.round(p_best[2],2))+"-"+str(np.round(np.exp(p_best[1]),2))+"e^{-"+str(np.round(p_best[0],2))+"\\sqrt{M}}$"
plt.plot(range(10, 181), [fit_fun(n_shots, *p_best) for n_shots in range(10, 181)], ls='--', lw=lw,
         color='0.8', zorder=-10, label=best_fit_label)
plt.xticks(n_shots_array)
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='lower right', fontsize=fs)
ax.tick_params(labelsize=fs*5/6)
plt.tight_layout()
plt.savefig('../wp_NK/mitigation_plots/ionq_mitigation.pdf')
plt.show()
# -

df.reset_index(level=0, inplace=True, drop=True)
df.to_pickle('mitigated_hardware_matrices.pkl')




