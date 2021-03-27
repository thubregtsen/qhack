import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import sys
import numpy as np
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

noiseless_kernel_matrix = np.load('noiseless_kernel_matrix.npy')


def calc_alignment(A, B):
    AB = _matrix_inner_product(A, B)
    AA = _matrix_inner_product(A, A)
    BB = _matrix_inner_product(B, B)
    return(BB/(AA**0.5 * BB ** 0.5))


def _matrix_inner_product(A, B):
    """Frobenius/Hilbert-Schmidt inner product between two matrices
    Args:
        A (array[float]): First matrix, assumed to be a square array.
        B (array[float]): Second matrix, assumed to be a square array.
    Returns:
        float: Inner product of A and B
    """
    return np.trace(np.dot(np.transpose(A), B))


def sample_from_measurement_distribution(distribution, shots):
    values = list(distribution.keys())
    probabilities = list(distribution.values())
    if sum(probabilities) != 1:

        probabilities_added = (1-sum(probabilities))/len(probabilities)
        for i in range(len(probabilities)):
            probabilities[i] += probabilities_added

    samples = np.random.choice(values, n_shots, p=probabilities)
    kernel_entry = 0
    for i in samples:
        if i == '000':
            kernel_entry += 1
    return(kernel_entry/shots)


def translate_folder(module_path, n_shots):
    print(n_shots, 'shots')
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
    kernel_array = []
    n_shots_array = [15, 25, 50, 75, 125, 150, 175]  # , 200, 500]
    kernel_matrices = []
    for n_shots in n_shots_array:
        kernel_array = [0] * 1830
        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_0_680/'))
        kernel_array[:679] = translate_folder(module_path, n_shots)

        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_681_929'))
        kernel_array[681:681+248] = translate_folder(module_path, n_shots)

        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_929_1229'))
        kernel_array[929:929+300] = translate_folder(module_path, n_shots)

        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_1229_1529'))
        kernel_array[1229:1529] = translate_folder(module_path, n_shots)

        module_path = os.path.abspath(
            os.path.join('./ionq_kernel_matrix_1529_1829'))
        kernel_array[1530:1830] = translate_folder(module_path, n_shots)

        N_datapoints = 60
        kernel_matrix = np.zeros((60, 60))
        index = 0
        for i in range(N_datapoints):
            for j in range(i, N_datapoints):
                kernel_matrix[i, j] = kernel_array[index]
                kernel_matrix[j, i] = kernel_matrix[i, j]
                index += 1
        kernel_matrix = np.reshape(kernel_matrix, (60, 60))
        alignment = calc_alignment(
            kernel_matrix, noiseless_kernel_matrix)
        print(alignment, 'alignment')
     #   kernel_matrices.append(kernel_matrix)

    # visualize_kernel_matrices(kernel_matrices)
