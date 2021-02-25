# -*- coding: utf-8 -*-
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

# # Quantum Embedding Kernels with PennyLane's kernels module
#
# _Authors: Peter-Jan Derks, Paul Fährmann, Elies Gil-Fuster, Tom Hubregtsen, Johannes Jakob Meyer and David Wierichs_
#
# Kernel methods are one of the cornerstones of classical machine learning. To understand what a kernel method does we first look at one of the possibly simplest methods to assign class labels to datapoints: linear classification.
#
# **TODO: Add intuitive explanation of kernel methods as in the paper**
#
# In this work, we will be concerned with _Quantum Embedding Kernels (QEKs)_, i.e. kernels that arise from embedding data into a quantum state. We formalize this by considering a quantum circuit $U(\boldsymbol{x})$ that embeds the datapoint $\boldsymbol{x}$ into the state
#
# $$
# |\psi(\boldsymbol{x})\rangle = U(\boldsymbol{x}) |0 \rangle.
# $$
#
# The kernel value is then given by the _overlap_ of the associated embedded quantum states
#
# $$
# k(\boldsymbol{x}, \boldsymbol{y}) = | \langle\psi(\boldsymbol{x})|\psi(\boldsymbol{y})\rangle|^2.
# $$

# ## A toy problem
#
# In this demonstration, we will treat a toy problem that showcases the inner workings of our approach. We will create the `DoubleCake` dataset. To do so, we first have to do some imports:

import pennylane as qml
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class DoubleCake:
    def _make_circular_data(self):    
        center_indices = np.array(range(0, self.num_sectors))
        sector_angle = 2*np.pi / self.num_sectors
        angles = (center_indices + 0.5) * sector_angle
        x = 0.7 * np.cos(angles)
        y = 0.7 * np.sin(angles)
        labels = 2 * np.remainder(np.floor_divide(angles, sector_angle), 2)- 1 
        
        return x, y, labels

    def __init__(self, num_sectors):
        self.num_sectors = num_sectors
        
        x1, y1, labels1 = self._make_circular_data()
        x2, y2, labels2 = self._make_circular_data()

        # x and y coordinates of the datapoints
        self.x = np.hstack([x1, .5 * x2])
        self.y = np.hstack([y1, .5 * y2])
        
        # Canonical form of dataset
        self.X = np.vstack([self.x, self.y]).T
        
        self.labels = np.hstack([labels1, -1 * labels2])
        
        # Canonical form of labels
        self.Y = self.labels.astype(int)

    def plot(self, ax, show_sectors=False):
        ax.scatter(self.x, self.y, c=self.labels, cmap=mpl.colors.ListedColormap(['#FF0000', '#0000FF']), s=10)
        sector_angle = 360/self.num_sectors
        
        if show_sectors:
            for i in range(self.num_sectors):
                color = ['#FF0000', '#0000FF'][(i % 2)]
                other_color = ['#FF0000', '#0000FF'][((i + 1) % 2)]
                ax.add_artist(mpl.patches.Wedge((0, 0), 1, i * sector_angle, (i+1)*sector_angle, lw=0, color=color, alpha=0.1, width=.5))
                ax.add_artist(mpl.patches.Wedge((0, 0), .5, i * sector_angle, (i+1)*sector_angle, lw=0, color=other_color, alpha=0.1))
                ax.set_xlim(-1, 1)

        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.axis("off")


# Let's now have a look at our dataset. In our example, we will work with 6 sectors:

# +
dataset = DoubleCake(6)

dataset.plot(plt.gca(), show_sectors=True)
# -

# ## Defining a Quantum Embedding Kernel
#
# PennyLane's `kernels` module allows for a particularly simple implementation of Quantum Embedding Kernels. To be able to use it we first import some useful packages:

import pennylane as qml
