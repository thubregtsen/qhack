import pandas as pd
import json
import os
import sys
import numpy as np
import pennylane as qml
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# +
# load data 

noiseless_kernel_matrix = np.load('noiseless_kernel_matrix.npy')


# -

def sample_from_measurement_distribution(distribution, n_shots):
    values = list(distribution.keys())
    probabilities = np.array(list(distribution.values()))
    sum_probabilities = np.sum(probabilities)
    if sum_probabilities != 1:
        probabilities += (1-sum_probabilities)/len(probabilities)

    samples = np.random.choice(values, n_shots, p=probabilities)
    kernel_entry = np.sum(samples == '000')/n_shots

    return(kernel_entry)


def translate_folder(module_path, n_shots):
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


def build_kernel_matrix(kernel_array):
    N_datapoints = 60
    index = 0
    kernel_matrix = np.zeros((60, 60))
    for i in range(N_datapoints):
        for j in range(i, N_datapoints):
            kernel_matrix[i, j] = kernel_array[index]
            kernel_matrix[j, i] = kernel_matrix[i, j]
            index += 1
    return(kernel_matrix)


if __name__ == "__main__":
    n_shots_array = [15, 25, 50, 75, 100, 125, 150, 175]
    df = pd.DataFrame()
    for n_shots in n_shots_array:
        kernel_array = [0] * 1830

        module_path = os.path.abspath(
            os.path.join('./data/ionq_kernel_matrix_0_679/'))
        kernel_array[:679] = translate_folder(module_path, n_shots)
        module_path = os.path.abspath(
            os.path.join('./data/ionq_kernel_matrix_679_680/'))
        kernel_array[679:681] = translate_folder(module_path, n_shots)

        module_path = os.path.abspath(
            os.path.join('./data/ionq_kernel_matrix_681_929'))
        kernel_array[681:681+248] = translate_folder(module_path, n_shots)

        module_path = os.path.abspath(
            os.path.join('./data/ionq_kernel_matrix_929_1229'))
        kernel_array[929:1229] = translate_folder(module_path, n_shots)

        module_path = os.path.abspath(
            os.path.join('./data/ionq_kernel_matrix_1229_1529'))
        kernel_array[1229:1529] = translate_folder(module_path, n_shots)

        module_path = os.path.abspath(
            os.path.join('./data/ionq_kernel_matrix_1529'))
        kernel_array[1529] = translate_folder(module_path, n_shots)

        module_path = os.path.abspath(
            os.path.join('./data/ionq_kernel_matrix_1529_1829'))
        kernel_array[1530:1830] = translate_folder(module_path, n_shots)

        kernel_matrix = build_kernel_matrix(kernel_array)
        alignment = qml.kernels.matrix_inner_product(kernel_matrix, noiseless_kernel_matrix, normalize=True)
        print(alignment)
        df = df.append({
            'n_shots': n_shots,
            'kernel_matrix': kernel_matrix,
            'alignment': alignment,
            'pipeline': 'No post-processing'
        }, ignore_index=True)

    df.to_pickle('hardware_matrices.pkl')




