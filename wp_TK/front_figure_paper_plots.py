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
        X_dummy = np.load(f)
        y_dummy_label = np.load(f)
        y_dummy = np.load(f)
        y_dummy_random_label = np.load(f)
        y_dummy_random = np.load(f)
        X_train = np.load(f)
        y_train = np.load(f)
        X_test = np.load(f)
        y_test = np.load(f)
        
    return X_dummy, y_dummy_label, y_dummy, X_train, y_train, X_test, y_test, y_dummy_random_label, y_dummy_random
    
def plot_classification(ax, X_dummy, y_dummy_label, y_dummy, X_train, y_train, X_test, y_test, markersize=15, marker="o", clip=1, no_test=False):
    
    xx, yy = np.meshgrid(np.unique(X_dummy[:,0]), np.unique(X_dummy[:,1]))
    zz = np.zeros_like(xx)

    for idx in np.ndindex(*xx.shape):
        zz[idx] = y_dummy[np.intersect1d((X_dummy[:,0] == xx[idx]).nonzero(), (X_dummy[:,1] == yy[idx]).nonzero())[0]]


    ax.contourf(xx, yy, np.clip(zz, -clip, clip), cmap=linmap, alpha=0.5)
    ax.scatter(X_train[np.where(y_train == 1)[0],0], X_train[np.where(y_train == 1)[0],1], color="C0", facecolors=None, marker=marker, s=markersize, label="Train")
    ax.scatter(X_train[np.where(y_train == -1)[0],0], X_train[np.where(y_train == -1)[0],1], color="C1", facecolors=None, marker=marker, s=markersize, label="Train")
    if not no_test:
        ax.scatter(X_test[np.where(y_test == 1)[0],0], X_test[np.where(y_test == 1)[0],1], color="C0", facecolors="none", marker=marker, s=markersize, label="Test")
        ax.scatter(X_test[np.where(y_test == -1)[0],0], X_test[np.where(y_test == -1)[0],1], color="C1", facecolors="none", marker=marker, s=markersize, label="Test")
    # ax.set_ylim([0, 1])
    # ax.set_xlim([0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=[
            mpl.lines.Line2D([0], [0], color="w", markeredgecolor="black", markerfacecolor="black", marker=marker, markersize=np.sqrt(markersize), label='Train'),
            mpl.lines.Line2D([0], [0], color="w", markeredgecolor="black", marker=marker, markersize=np.sqrt(markersize), label='Test'),],  
        bbox_to_anchor=[0.5, 0.96], 
        loc='lower center', ncol=2, frameon=False)
# -

fmt = rsmf.setup(r"\documentclass[twocolumn,superscriptaddress,nofootinbib]{revtex4-2}")

# +
fig = fmt.figure(aspect_ratio=0.8)

X_dummy, y_dummy_label, y_dummy, X_train, y_train, X_test, y_test, y_dummy_random_label, y_dummy_random = load_data_elies("dataset_checkerboard.npy")

plot_classification(plt.gca(), X_dummy, y_dummy_label, y_dummy, X_train, y_train, X_test, y_test, clip=0.8, no_test=True)
plt.gca().get_legend().remove()
plt.gca().axis("off")
plt.tight_layout()
plt.savefig("after.pdf")

# +
fig = fmt.figure(aspect_ratio=0.8)

X_dummy, y_dummy_label, y_dummy, X_train, y_train, X_test, y_test, y_dummy_random_label, y_dummy_random = load_data_elies("dataset_checkerboard.npy")

plot_classification(plt.gca(), X_dummy, y_dummy_random_label, y_dummy_random, X_train, y_train, X_test, y_test, clip=0.8, no_test=True)
plt.gca().get_legend().remove()
plt.gca().axis("off")
plt.tight_layout()
plt.savefig("before.pdf")

# +
fig = fmt.figure()

plot_classification(plt.gca(), *load_data_tom("dataset_checkerboard.npy"))


plt.tight_layout()
# -
glob.glob("*.npy")


# +
fig=fmt.figure(aspect_ratio=1/5.0)

a = [.087]

for i in range(400):
    t = (1/(100 + i**1.3))
    
    a.append(a[-1] + np.random.uniform(-t * (i / 400), t))

    
plt.plot(a, lw=2.0, color="#8288B9")

ax = plt.gca()

ax.set_xticks([])
ax.set_yticks([])
ax.axis("off")
plt.savefig("ta_graph.pdf")
# -
with open("alignment_checkerboard.npy", 'rb') as f:
    alignment = np.load(f)


plt.plot(alignment)

# +
fig=fmt.figure(aspect_ratio=1/5.0)

    
plt.plot(alignment, lw=2.0, color="#8288B9")

ax = plt.gca()

ax.set_xticks([])
ax.set_yticks([])
ax.axis("off")
plt.savefig("ta_graph.pdf")
# -


