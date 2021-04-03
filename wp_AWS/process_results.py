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


def translate_folder(module_path):
    index = 1
    data_dict = {'timestamp': [], 'measurement_result': []}
    for dirs, subdir, files in os.walk(module_path):

        # print(subdir)
        for file in files:
            # print(dirs + '/' + file)
            if file == 'results.json':
                with open(dirs + '/' + file) as myfile:
                    data = myfile.read()
                obj = json.loads(data)
                timestamp = obj['taskMetadata']['createdAt']
                try:
                    measurement_result = obj['measurementProbabilities']['000']
                except KeyError:
                    measurement_result = 0
                time_for_sorting = timestamp[8:10] + \
                    timestamp[11:13] + timestamp[14:16] + timestamp[17:19]

                data_dict['timestamp'].append(time_for_sorting)
                data_dict['measurement_result'].append(measurement_result)

                index += 1
    print(index, 'index')
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
            ax=ax[i],  # [i],
            cmap='Spectral',
            cbar=False
        )
    if draw_last_cbar:
        ch = plot.get_children()
        fig.colorbar(ch[0], ax=ax[-2], cax=ax[-1])
    plt.show()


if __name__ == "__main__":
    n_shots=175
    kernel_array = [0] * 1830

    module_path = os.path.abspath(
        os.path.join('./ionq_kernel_matrix_0_679/'))
    kernel_array[:679] = translate_folder(module_path)
    module_path = os.path.abspath(
        os.path.join('./ionq_kernel_matrix_679_680/'))
    kernel_array[679:681] = translate_folder(module_path)

    module_path = os.path.abspath(
        os.path.join('./ionq_kernel_matrix_681_929'))
    kernel_array[681:681+248] = translate_folder(module_path)

    module_path = os.path.abspath(
        os.path.join('./ionq_kernel_matrix_929_1229'))
    kernel_array[929:1229] = translate_folder(module_path)

    module_path = os.path.abspath(
        os.path.join('./ionq_kernel_matrix_1229_1529'))
    kernel_array[1229:1529] = translate_folder(module_path)

    module_path = os.path.abspath(
        os.path.join('./ionq_kernel_matrix_1529'))
    kernel_array[1529] = translate_folder(module_path)


    module_path = os.path.abspath(
        os.path.join('./ionq_kernel_matrix_1529_1829'))
    kernel_array[1530:1830] = translate_folder(module_path)
    index=0
    N_datapoints = 60
    
    kernel_matrix = np.zeros((60, 60))

    #print(kernel_array[1529])

    for i in range(N_datapoints):
        for j in range(i, N_datapoints):
            kernel_matrix[i, j] = kernel_array[index]
            kernel_matrix[j, i] = kernel_matrix[i, j]
            index += 1
    kernel_matrix = np.reshape(kernel_matrix, (60, 60))
    np.save('ionq_hardware_kernel_matrix', kernel_matrix)
    visualize_kernel_matrices([kernel_matrix])
    print(kernel_matrix)

noiseless_kernel_matrix = np.load('../wp_NK/testing_ionq_classical.npy')
print(noiseless_kernel_matrix)

# +
unpickled_df = pd.read_pickle('mitigated_hardware_matrices.pkl')
print(unpickled_df)

mitigated_kernel_matrix = unpickled_df['kernel_matrix'][519]
print(unpickled_df['alignment'][519])
print(mitigated_kernel_matrix)
# -

visualize_kernel_matrices([kernel_matrix, mitigated_kernel_matrix, noiseless_kernel_matrix])




