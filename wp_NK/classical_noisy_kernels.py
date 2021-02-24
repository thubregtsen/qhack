# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
import numpy as np
import torch
from torch.nn.functional import relu

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor

import matplotlib.pyplot as plt

import jupytext
import timeit
np.random.seed(42)

# -

# # Tom's Data

# +
# load the data
X, y = load_iris(return_X_y=True)

print("The dataset contains X and y, each of length", len(X))
print("X contains", len(X[0]), "features")
print("y contains the following classes", np.unique(y))

# pick inputs and labels from the first two classes only,
# corresponding to the first 100 samples
# -> meanig y now consists of 2 classes: 0, 1; still stored in order, balanced 50:50
X = X[:100, 0:2]
y = y[:100]

print("The dataset is trimmed so that the total number of samples are ", len(X))
print(
    "The original tutorial sticked with 4 features, I (Tom) reduced it to ", len(X[0]))

# scaling the inputs is important since the embedding we use is periodic
# -> data is scaled to np.min(X)=-2.307; np.max(X)= 2.731
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

print("X is normalized to the range", np.min(X_scaled), np.max(X_scaled))

# scaling the labels to -1, 1 is important for the SVM and the
# definition of a hinge loss
# -> now making the 2 classes: -1, 1
y_scaled = 2 * (y - 0.5)
print("y is normalized to drop a class, and now contains", np.sum(
    [1 if x == -1 else 0 for x in y_scaled]), "\"-1\" classes and ", np.sum([1 if x == 1 else 0 for x in y_scaled]), "\"1\" classes")

# -> result of train_test_split:
# len(X_train)=75, 39 labelled 1, 36 labelled -1
# len(X_test)=25
# data is shuffled prior to split (shuffled variable in train_test_split is default True)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)
print("Lastly, the data is shuffled and split into", len(
    X_train), "training samples and", len(X_test), "samples")
# -

#

# +

noise_strength_array= np.arange(0,100,1)
score_array = []
runtime_array = []

for noise_strength in noise_strength_array:
    start = timeit.default_timer()

    def my_kernel(X,Y):
        kernel = np.dot(X,Y.T)
    
        noise = np.random.normal(loc=0, scale=noise_strength,size=(75, 75))
        #print(noise,'noise')
        noise *= np.tri(*noise.shape, k=-1)
        noise += noise.transpose()
        #np.fill_diagonal(noise,1)
        return(kernel+noise)
    score = 0
    for i in range(50):
        
        # Create a SVC classifier using a linear kernel
        svm_lin = SVC(kernel=my_kernel, C=1, random_state=0)
        # Train the classifier
        svm_lin.fit(X_train, y_train)

        score += svm_lin.score(X_train,y_train)
    stop = timeit.default_timer()
    runtime_array.append(stop-start)
    score_array.append(score/50)
    

# -

# # Noise model:
# The probability density for the Gaussian distribution is
#
# $p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}
# e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} },$
#
# where $\mu$ is the mean and $\sigma$ the standard deviation
#
# In my noise model I set $\mu=0$ and vary $\sigma$
#
#
#

# +
fig, ax = plt.subplots()
ax.plot(noise_strength_array, score_array)
ax.set_xlabel('Noise')
ax.set_ylabel('Score')

fig, ax = plt.subplots()
ax.plot(noise_strength_array,runtime_array)
ax.set_xlabel('Noise')
ax.set_ylabel('runtime')
# -


