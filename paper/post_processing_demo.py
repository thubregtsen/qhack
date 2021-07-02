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
import itertools
import numpy as np
import pennylane as qml

# Plotting
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# -

# # A noisy kernel matrix
# The matrix we import from the appended data file (`noisy_kernel_matrix`) was computed with a simulated depolarizing noise model (base noise constant 0.98; one-qubit depolarizing channel after each gate, including idling gates; depolarizing rate scaled with rotation angle) and a limited shot budget of 1000 shots per kernel matrix entry. As you can see, we computed the diagonal entries although we know their exact value to be 1. This is because we will use them for noise mitigation.
# We also import the simulated noiseless kernel matrix (`exact_kernel_matrix`) to evaluate our results later on.

num_wires = 5
from data.mitigation_demo_data import noisy_kernel_matrix, exact_kernel_matrix
print(f"We computed the diagonal (would be 1 analytically) for mitigation:\n{np.diag(noisy_kernel_matrix)}")

# # Post-processing functions: regularization and mitigation
# Let us now collect the post-processing methods presented in the paper. We consider three regularization and mitigation methods respectively.
#
# All three regularization methods are implemented in our contribution to PennyLane, the `qml.kernels` module.method
#
# As for the mitigation methods, `mitigate_depolarizing_noise` from `qml.kernels` is used with three different `method` keyword arguments corresponding to the three concepts from the paper.
#
# We also define a "do-nothing" function `Id` to skip steps in the pipelines below.

# +
r_Tikhonov = qml.kernels.displace_matrix
r_thresh = qml.kernels.threshold_matrix
r_SDP = qml.kernels.closest_psd_matrix

m_single = lambda mat: qml.kernels.mitigate_depolarizing_noise(mat, num_wires, method='single')
m_mean = lambda mat: qml.kernels.mitigate_depolarizing_noise(mat, num_wires, method='average')
m_split = lambda mat: qml.kernels.mitigate_depolarizing_noise(mat, num_wires, method='split_channel')

Id = lambda mat: mat
# -

# ## Setting up post-processing pipelines
# Now we combine the methods above into pipelines with three steps. We will use the ordering _regularize_-_mitigate_-_regularize_ and allow to skip any of these steps by including `Id` as an option. 
#
# We also define a simple function `apply_pipeline` that applies all functions from a list to an input matrix successively. In case of instabilities in the SDP (in `r_SDP`) or input matrices for which our noise rate estimation (in `mitigate_depolarizing_noise`) is not well-defined, we make sure the function exits properly.

# +
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


# -

# ## Applying the pipelines to the noisy kernel matrix
# Now we simply iterate over all pipelines and apply them to the noisy kernel matrix from above. 
#
# We will *filter* the pipelines for duplicates/redundant methods:
# 1. We store the results in a dictionary and generate the key with the function names but ignoring `Id`. This makes the key for e.g. the pipelines `(Id, Id, r_thresh)` and `(r_thresh, Id, Id)` the same and we do not compute both of them, as they are equivalent. 
#
# 2. If we use `r_SDP` as first method, the mitigation methods will estimate a noise rate of zero because the diagonal entries of the output of `r_SDP` are fixed to one, the noiseless value. 
#
# 3. If we do not mitigate, the first regularization step will already yield a positive semi-definite matrix that is passed to the second regularziation, which then only will modify the matrix further if it fixes the diagonal to one (`r_SDP`) and the first regularization did not.
#
# If you want to skip the filtering, simply deactivate the `filter` boolean variable in the next field.
#

filter = True
mitigated_kernel_matrices = {}
for pipeline in pipelines:
    if filter:
        key = ', '.join([function_names[function] for function in pipeline if function!=Id])
        if key=='': # Give the Id-Id-Id pipeline a proper name
            key = 'No post-processing'
        if key in mitigated_kernel_matrices.keys(): # Skip duplicated keys (the dict would be overwritten anyways)
            continue
        if pipeline[0]==r_SDP and (pipeline[1]!=Id or pipeline[2]!=Id): # Skip r_SDP - ~ID - ~Id
            continue
        if pipeline[1]==Id and pipeline[2] in [r_Tikhonov, r_thresh]: # Skip regularize - Id - r_Tikhonov/thresh
            continue
    else:
        key = ', '.join([function_names[function] for function in pipeline])
        
    mitigated_kernel_matrices[key] = apply_pipeline(pipeline, noisy_kernel_matrix)

# ### How many pipelines are we left with?
# Originally, `pipelines` contained $4^3=64$ combinations.
# Our filtering for duplicates above reduced this number to $42$ pipelines, including the pipeline `Id`-`Id`-`Id`.
#
# In case you deactivated the filter, you will find that out of the $64$ mitigated matrices there are only $42$ unique matrices (with a precision of up to $10^{-8}$ per matrix entry:

print(f"# Pipelines: {len(pipelines)}")
print(f"# Filtered pipelines: {len(mitigated_kernel_matrices)}")
unique_matrices = np.unique([np.round(mat, 8) for mat in mitigated_kernel_matrices.values()], axis=0)
print(f"# Unique matrices: {len(unique_matrices)}")

# ## Evaluating the results
# We can use the `matrix_inner_product` function from `qml.kernels` with the option `normalize=True` to compute the alignment between the exact, noiseless matrix and the various mitigated matrices (including the pipeline `Id`-`Id`-`Id`, i.e. the original noisy matrix)

# %matplotlib notebook
sorted_keys = sorted(list(mitigated_kernel_matrices.keys()))
sorted_matrices = [mitigated_kernel_matrices[key] for key in sorted_keys]
alignments = [
    (
        qml.utils.frobenius_inner_product(exact_kernel_matrix, mitigated, normalize=True).item()
        if mitigated is not None
        else None
    )
    for mitigated in sorted_matrices
]
df = pd.DataFrame({
    'pipeline': sorted_keys,
    'mitigated_matrix': sorted_matrices,
    'alignment': alignments,
})
plot = sns.catplot(data=df, x='pipeline', y='alignment', aspect=1.8)
plt.setp( plt.gca().xaxis.get_majorticklabels(), rotation=90 )
plt.xlabel('')
plt.ylabel('Alignment $\operatorname{A}(\overline{K},K)$')
plt.ylim((df.alignment.min()-0.05, 1.))
plt.tight_layout()

# ## Conclusion
# The plot above shows the alignment of the mitigated matrices and the noiseless matrix, which we show as color-coded plots in our paper. For the specific noise parameters ($1-\lambda_0=0.02,\;M=1000$) we find the pipeline $[\mathit{m}_\mathrm{mean}, r_\mathrm{SDP}]$ to be best (c.f. Figure 10 in the paper, label "4"):

df.loc[df.alignment.idxmax()].pipeline


