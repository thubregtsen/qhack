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
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import rsmf

linmap = mpl.colors.LinearSegmentedColormap.from_list("test", ["C1", "white" ,"C0"], N=1024)# mpl.colors.ListedColormap(["C1", "C0"])

import rsmf
### !!! you need to reference the main.tex !!! ####
fmt = rsmf.setup("style/main.tex")


# +
def load_data_tom(filename):
    with open(filename, 'rb') as f:
        X_dummy = np.load(f)
        y_dummy = np.load(f)
        X_train = np.load(f)
        y_train = np.load(f)
        X_test = np.load(f)
        y_test = np.load(f)
        
    return X_dummy, y_dummy, y_dummy, X_train, y_train, X_test, y_test

def load_data_elies(filename):
    with open(filename, 'rb') as f:
        X_dummy_c = np.load(f)
        y_dummy__random_c = np.load(f)
        y_dummy_random_real_c = np.load(f)
        y_dummy_c = np.load(f)
        y_dummy_real_c = np.load(f)
        X_train_c = np.load(f)
        y_train_c = np.load(f)
        X_test_c = np.load(f)
        y_test_c = np.load(f)
        
    return X_dummy_c, y_dummy__random_c, y_dummy_random_real_c, y_dummy_c, y_dummy_real_c, X_train_c, y_train_c, X_test_c, y_test_c

    
def plot_classification(ax, X_dummy, y_dummy_label, y_dummy, X_train, y_train, X_test, y_test, markersize=15, marker="o", clip=1):
    
    xx, yy = np.meshgrid(np.unique(X_dummy[:,0]), np.unique(X_dummy[:,1]))
    zz = np.zeros_like(xx)

    for idx in np.ndindex(*xx.shape):
        zz[idx] = y_dummy[np.intersect1d((X_dummy[:,0] == xx[idx]).nonzero(), (X_dummy[:,1] == yy[idx]).nonzero())[0]]


    ax.contourf(xx, yy, np.clip(zz, -clip, clip), cmap=linmap, alpha=0.5)
    ax.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="C0", facecolors=None, marker=marker, s=markersize, label="Train")
    ax.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="C1", facecolors=None, marker=marker, s=markersize, label="Train")
    ax.scatter(X_test[np.where(y_test == 1)[0],0], X_test[np.where(y_test == 1)[0],1], color="C0", facecolors="none", marker=marker, s=markersize, label="Test")
    ax.scatter(X_test[np.where(y_test == -1)[0],0], X_test[np.where(y_test == -1)[0],1], color="C1", facecolors="none", marker=marker, s=markersize, label="Test")
    ax.set_xticks([])
    ax.set_yticks([])
# -







# +
fig = fmt.figure(width_ratio=1., aspect_ratio=1.6)

counter = 421

markersize=15
marker="o"
clip=1
legend_offset = -.4

# checkerboard
## untrained
X_dummy_c, y_dummy__random_c, y_dummy_random_real_c, y_dummy_c, y_dummy_real_c, X_train_c, y_train_c, X_test_c, y_test_c = load_data_elies("./data/dataset_checkerboard.npy")
ax = fig.add_subplot(counter)
plot_classification(ax, X_dummy_c, y_dummy__random_c, y_dummy_random_real_c, X_train_c, y_train_c, X_test_c, y_test_c)
ax.set_ylabel("checkerboard")
counter += 1
## trained
ax = fig.add_subplot(counter)
plot_classification(ax, X_dummy_c, y_dummy_c, y_dummy_real_c, X_train_c, y_train_c, X_test_c, y_test_c)
counter += 1

# donuts
## untrained
X_dummy_c, y_dummy_c, y_dummy_real_c, y_dummy__random_c, y_dummy_random_real_c, X_train_c, y_train_c, X_test_c, y_test_c = load_data_elies("./data/dataset_symmetricdonuts.npy")
ax = fig.add_subplot(counter)
plot_classification(ax, X_dummy_c, y_dummy__random_c, y_dummy_random_real_c, X_train_c, y_train_c, X_test_c, y_test_c)
ax.set_ylabel("donuts")
counter += 1
## trained
ax = fig.add_subplot(counter)
plot_classification(ax, X_dummy_c, y_dummy_c, y_dummy_real_c, X_train_c, y_train_c, X_test_c, y_test_c)
counter += 1

# zero vs non-zero
## untrained
ax = fig.add_subplot(counter)
plot_classification(ax, *load_data_tom("./data/dataset_MNIST_23_zero_untrained.npy"))
ax.set_ylabel("zero vs non-zero")
counter += 1
## trained
ax = fig.add_subplot(counter)
plot_classification(ax, *load_data_tom("./data/dataset_MNIST_23_zero_trained.npy"))
counter += 1

# one vs one-zero
## untrained
ax = fig.add_subplot(counter)
plot_classification(ax, *load_data_tom("./data/dataset_MNIST_23_one_untrained.npy"))
ax.set_xlabel("untrained")
ax.set_ylabel("one vs non-one")
ax.legend(
    handles=[
        mpl.lines.Line2D([0], [0], color="w", markeredgecolor="C0", markerfacecolor=None, marker=marker, markersize=np.sqrt(markersize), label='train -1'),
        mpl.lines.Line2D([0], [0], color="w", markeredgecolor="C0", markerfacecolor="C0", marker=marker, markersize=np.sqrt(markersize), label='test -1'),], 
    bbox_to_anchor=[0.465, legend_offset], 
    loc='lower center', ncol=4, frameon=False, handletextpad=0.00, columnspacing=0.0, borderpad=0.05) # borderpad=0
counter += 1
# trained
ax = fig.add_subplot(counter)
plot_classification(ax, *load_data_tom("./data/dataset_MNIST_23_one_trained.npy"))
ax.set_xlabel("trained")
ax.legend(
    handles=[
        mpl.lines.Line2D([0], [0], color="w", markeredgecolor="C1", markerfacecolor=None, marker=marker, markersize=np.sqrt(markersize), label='train 1'),
        mpl.lines.Line2D([0], [0], color="w", markeredgecolor="C1", markerfacecolor="C1", marker=marker, markersize=np.sqrt(markersize), label='test 1'),],  
    bbox_to_anchor=[0.465, legend_offset], 
    loc='lower center', ncol=4, frameon=False, handletextpad=0.00, columnspacing=0.0, borderpad=0.05) # borderpad=0


counter += 1

plt.tight_layout()
#plt.show()
plt.savefig("./images/numerics_plots_trainable_kernel.jpg")
