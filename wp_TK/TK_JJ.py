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

import pennylane as qml
from pennylane import numpy as np
import tk_lib as tk
from cake_dataset import Dataset as Cake
from cake_dataset import DoubleCake
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm

# +
dataset = DoubleCake(0, 6)
dataset.plot(plt.gca())

X = np.vstack([dataset.X, dataset.Y]).T
Y = dataset.labels_sym.astype(int)


# +
def layer(x, params, wires, i0=0, inc=1):
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])
        
    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

@qml.template
def ansatz(x, params, wires):
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))
        
def random_params(num_wires, num_layers):
    return np.random.uniform(0, 2*np.pi, (num_layers, 2, num_wires))


# -

dev = qml.device("lightning.qubit", wires=5)
wires = list(range(5))
W = np.random.normal(0, .7, (2, 30))
k = qml.kernels.EmbeddingKernel(lambda x, params: ansatz(x @ W, params, wires), dev)

init_params = random_params(5, 6)

k([0.1, 0.2], [0.2, 0.3], init_params)

svm = tk.train_svm(k, X, Y, init_params)

svm.predict(X)

# +
xx = np.linspace(-1, 1, 14)
yy = np.linspace(-1, 1, 14)

XX, YY = np.meshgrid(xx,yy)

ZZ = np.zeros_like(XX)
for idx in tqdm.notebook.tqdm(np.ndindex(*XX.shape)):
    ZZ[idx] = svm.predict(np.array([XX[idx], YY[idx]])[np.newaxis,:])
    
plot_data_init = {'XX' : XX, 'YY' : YY, 'ZZ' : ZZ}
    
plt.contourf(XX, YY, ZZ, cmap=mpl.colors.ListedColormap(['#FF0000', '#0000FF']), alpha=.2, levels=[-1, 0,  1])
dataset.plot(plt.gca())
# -



params = init_params

# +
opt = qml.GradientDescentOptimizer(1.5)
#params = hist[-1]
hist = []

for i in tqdm.notebook.tqdm(range(100)):
    hist.append(params)
    subset = np.random.choice(list(range(len(X))), 4)
    params = opt.step(lambda _params: -k.target_alignment(X[subset], Y[subset], _params), params)
    if i % 10 == 0:
        print(i, " - Alignment = ", k.target_alignment(X, Y, params))
# -

svm2 = tk.train_svm(k, X, Y, params)

# +
xx = np.linspace(-1, 1, 14)
yy = np.linspace(-1, 1, 14)

XX, YY = np.meshgrid(xx,yy)

ZZ = np.zeros_like(XX)
for idx in tqdm.notebook.tqdm(np.ndindex(*XX.shape)):
    ZZ[idx] = svm2.predict(np.array([XX[idx], YY[idx]])[np.newaxis,:])
    
plot_data_trained = {'XX' : XX, 'YY' : YY, 'ZZ' : ZZ}
    
plt.contourf(XX, YY, ZZ, cmap=mpl.colors.ListedColormap(['#FF0000', '#0000FF']), alpha=.2, levels=[-1, 0,  1])
dataset.plot(plt.gca())
# -

plt.contourf(XX, YY, ZZ, cmap=mpl.colors.ListedColormap(['#FF0000', '#0000FF']), alpha=.2, levels=[-1, 0,  1])
dataset.plot(plt.gca())

ZZ


