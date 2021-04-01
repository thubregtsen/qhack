import scipy as sp
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
formatter = rsmf.setup(
    r"\documentclass[twocolumn,superscriptaddress,nofootinbib]{revtex4-2}")

# # load data

df = pd.read_pickle('hardware_matrices.pkl')
noiseless_kernel_matrix = np.load('noiseless_kernel_matrix.npy')
n_shots_array = df['n_shots']

# ## Define pipelines

num_wires = 3
r_Tikhonov = qml.kernels.displace_matrix
r_thresh = qml.kernels.threshold_matrix
r_SDP = qml.kernels.closest_psd_matrix


def m_single(mat): return qml.kernels.mitigate_depolarizing_noise(
    mat, num_wires, method='single')

def m_mean(mat): return qml.kernels.mitigate_depolarizing_noise(
    mat, num_wires, method='average')

def m_split(mat): return qml.kernels.mitigate_depolarizing_noise(
    mat, num_wires, method='split_channel')


def Id(mat): return mat


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
pipelines = list(itertools.product(
    regularizations, mitigations, regularizations))


def apply_pipeline(pipeline, mat):
    out = np.copy(mat)
    for function in pipeline:
        try:
            out = function(out)
            if np.any(np.isinf(out)):
                raise ValueError
        except Exception as e:  # Catches problems in the SDP and problems caused by matrices that are too noisy for mitigation
            print(e)
            return None

    return out


for n_shots in tqdm.notebook.tqdm(n_shots_array):
    noisy_kernel_matrix = df.loc[(df.n_shots == n_shots) & (
        df.pipeline == 'No post-processing')].kernel_matrix.item()
    used_pipelines = set(['No post-processing'])
    for pipeline in pipelines:

        if filter:
            key = ', '.join([function_names[function]
                             for function in pipeline if function != Id])
            if key == '':  # Give the Id-Id-Id pipeline a proper name
                key = 'No post-processing'
            # Skip duplicated keys (the dict would be overwritten anyways)
            if key in used_pipelines:
                continue
            # Skip r_SDP - ~ID - ~Id
            if pipeline[0] == r_SDP and (pipeline[1] != Id or pipeline[2] != Id):
                continue
            # Skip regularize - Id - r_Tikhonov/thresh
            if pipeline[1] == Id and pipeline[2] in [r_Tikhonov, r_thresh]:
                continue
        else:
            key = ', '.join([function_names[function]
                             for function in pipeline])
        mitigated_kernel_matrix = apply_pipeline(
            pipeline, noisy_kernel_matrix)
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

df.to_pickle('mitigated_hardware_matrices.pkl')
# -


