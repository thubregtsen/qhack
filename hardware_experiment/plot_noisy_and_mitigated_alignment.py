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

# ## load data

df = pd.read_pickle('mitigated_hardware_matrices.pkl')

# +
noisy_df = df.loc[df.pipeline == 'No post-processing']
mitigated_df = df.loc[df.pipeline != 'No post-processing']
num_top_pipelines = 1
n_shots_array = [15, 25, 50, 75, 100, 125, 150, 175]

def prettify_pipelines(x):
    funs = x.pipeline.split(', ')
    new_funs = []
    for fun in funs:
        if fun[0] == 'm':
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
    indices = mitigated_df.loc[mitigated_df.n_shots ==
                               n_shots].alignment.sort_values().index[-num_top_pipelines:]
    return mitigated_df.loc[indices]


best_df = pd.DataFrame()
for n_shots in tqdm.notebook.tqdm(n_shots_array):
    best_df = pd.concat([best_df, top_pipelines(n_shots)])
best_df['pretty_pipeline'] = best_df.apply(prettify_pipelines, axis=1)


p_noisy, pcov_noisy = sp.optimize.curve_fit(
    fit_fun, n_shots_array, noisy_df.alignment.to_numpy(), p0=[1, 0, 1])
p_best, pcov_best = sp.optimize.curve_fit(
    fit_fun, n_shots_array, best_df.alignment.to_numpy(), p0=[1, 0, 1])

# +
# %matplotlib notebook
fs = 14
ms = 50
lw = 2

sns.scatterplot(data=best_df, x='n_shots', y='alignment', hue='pretty_pipeline',
                style='pretty_pipeline', markers=['o', 'X'], s=ms
                )

sns.scatterplot(data=noisy_df, x='n_shots', y='alignment', color='k', marker='d', label='No post-processing',
                s=ms)


plt.xlabel(f"# Measurements $M$", fontsize=fs)
plt.ylabel("Alignment $\\operatorname{A}(\overline{K},K)$", fontsize=fs)
plt.plot([10, 180], [noisy_df.alignment.max()]*2,
         ls='-', lw=lw/2, color='0.8', zorder=-10)
noisy_fit_label = "$"+str(np.round(p_noisy[2], 2))+"-"+str(np.round(
    np.exp(p_noisy[1]), 2))+"e^{-"+str(np.round(p_noisy[0], 2))+"\\sqrt{M}}$"
plt.plot(range(10, 181), [fit_fun(n_shots, *p_noisy) for n_shots in range(10, 181)], ls=':', lw=lw,
         color='0.6', zorder=-10, label=noisy_fit_label)

best_fit_label = "$"+str(np.round(p_best[2], 2))+"-"+str(np.round(
    np.exp(p_best[1]), 2))+"e^{-"+str(np.round(p_best[0], 2))+"\\sqrt{M}}$"
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
