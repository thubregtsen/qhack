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
rnd_param = np.tensor([[[3.83185811, 2.43217597, 6.04150259, 6.10219181, 2.24859771],
         [2.10229161, 3.01695202, 0.65963585, 3.01146847, 3.09878739]],

        [[2.98450446, 4.67620615, 2.65282874, 0.27375408, 3.51592262],
         [4.42306178, 2.10907678, 1.9859467 , 3.15253185, 5.1835622 ]],

        [[3.15053375, 1.15141625, 6.26411875, 1.4094818 , 2.89303727],
         [0.88448723, 1.37280759, 1.42852862, 2.79908337, 4.82479853]],

        [[2.96944762, 2.92050829, 5.08902411, 4.38583442, 4.57381108],
         [2.87380533, 2.79339977, 5.40042108, 1.22715656, 3.55334794]],

        [[4.85217317, 2.32865449, 3.36674732, 5.37284552, 4.41718962],
         [5.46919267, 4.1238232 , 5.63482497, 1.35359693, 1.55163904]],

        [[4.7955417 , 1.71132909, 3.45214701, 1.30618948, 2.43551656],
         [5.99802411, 0.86416771, 1.52129757, 4.48878166, 5.1649024 ]]], requires_grad=True)


params = rnd_param

# +
opt = qml.GradientDescentOptimizer(2.)
#params = hist[-1]
hist = []

max_TA = -1e10
_optimal = None
for i in tqdm.notebook.tqdm(range(300)):
    hist.append(params)
    subset = np.random.choice(list(range(len(X))), 4)
    cost_fn = lambda _params: -k.target_alignment(X[subset], Y[subset], _params)
#     grad_fn = lambda _params: (-tk.target_alignment_grad(X[subset], Y[subset], k, _params), )
    params = opt.step(cost_fn, params)#, grad_fn =grad_fn)
    TA = k.target_alignment(X, Y, params)
    if TA>max_TA:
        max_TA = np.copy(TA)
        _optimal = np.copy(params)
    if i % 10 == 0:
        print(i, " - Alignment = ", TA)
# -

W

_optimal

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

# +
gs = mpl.gridspec.GridSpec(1, 3) 
fig = plt.figure(figsize=(10,3))

#Using the 1st row and 1st column for plotting heatmap

ax=plt.subplot(gs[0,0])
ax.contourf(plot_data_zero['XX'], plot_data_zero['YY'], plot_data_zero['ZZ'], cmap=mpl.colors.ListedColormap(['#FF0000', '#0000FF']), alpha=.2, levels=[-1, 0,  1])
ax.set_title("Zero, Target-Alignment = {:.3f}".format(k.target_alignment(X, Y, np.zeros_like(params))))
dataset.plot(ax)

ax=plt.subplot(gs[0,1])
ax.contourf(plot_data_init['XX'], plot_data_init['YY'], plot_data_init['ZZ'], cmap=mpl.colors.ListedColormap(['#FF0000', '#0000FF']), alpha=.2, levels=[-1, 0,  1])
dataset.plot(ax)
ax.set_title("Random, Target-Alignment = {:.3f}".format(k.target_alignment(X, Y, init_params)))

ax=plt.subplot(gs[0,2])
ax.contourf(plot_data_trained['XX'], plot_data_trained['YY'], plot_data_trained['ZZ'], cmap=mpl.colors.ListedColormap(['#FF0000', '#0000FF']), alpha=.2, levels=[-1, 0,  1])
ax.set_title("Trained, Target-Alignment = {:.3f}".format(k.target_alignment(X, Y, params)))
dataset.plot(ax)
plt.tight_layout()
# -

ZZ

# +
svm3 = tk.train_svm(k, X, Y, np.zeros_like(params))

xx = np.linspace(-1, 1, 14)
yy = np.linspace(-1, 1, 14)

XX, YY = np.meshgrid(xx,yy)

ZZ = np.zeros_like(XX)
for idx in tqdm.notebook.tqdm(np.ndindex(*XX.shape)):
    ZZ[idx] = svm3.predict(np.array([XX[idx], YY[idx]])[np.newaxis,:])
    
plot_data_zero = {'XX' : XX, 'YY' : YY, 'ZZ' : ZZ}
    
plt.contourf(XX, YY, ZZ, cmap=mpl.colors.ListedColormap(['#FF0000', '#0000FF']), alpha=.2, levels=[-1, 0,  1])
dataset.plot(plt.gca())
# -

k.target_alignment(X, Y, np.zeros_like(params))

k.probs_qnode.qtape


