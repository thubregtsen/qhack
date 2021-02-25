import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Wedge

class Dataset:
    def _make_data(self):    
        center_indices = np.random.randint(0, self.num_sectors, self.num_samples)
        sector_angle = 2*np.pi / self.num_sectors
        angles = (center_indices + 0.5) * sector_angle

        sub_angles = np.random.uniform(0, 2*np.pi, self.num_samples)
        r = 0.15 * np.vectorize(lambda u: u if u < 1 else 2-u)(np.random.uniform(0, 2, self.num_samples))
        dx = r * np.cos(sub_angles)
        dy = r * np.sin(sub_angles)

        X = 0.7 * np.cos(angles) + dx
        Y = 0.7 * np.sin(angles) + dy

        labels = np.remainder(center_indices, 2)
        labels_sym = 2 * np.remainder(np.floor_divide(angles, sector_angle), 2) - 1

        return X, Y, labels, labels_sym 

    def _make_simple_data(self):    
        center_indices = np.array(range(0, self.num_sectors))
        sector_angle = 2*np.pi / self.num_sectors
        angles = (center_indices + 0.5) * sector_angle
        X = 0.7 * np.cos(angles)
        Y = 0.7 * np.sin(angles)
        labels = np.remainder(center_indices, 2)
        labels_sym = 2 * np.remainder(np.floor_divide(angles, sector_angle), 2) - 1 
        return X, Y, labels, labels_sym

    def __init__(self, num_samples, num_sectors):
        self.num_samples = num_samples
        self.num_sectors = num_sectors
        if self.num_samples != 0:
            self.X, self.Y, self.labels, self.labels_sym = self._make_data()
        else:
            self.X, self.Y, self.labels, self.labels_sym = self._make_simple_data()

    def plot(self, ax):
        ax.scatter(self.X, self.Y, c=self.labels, cmap=ListedColormap(['#FF0000', '#0000FF']), s=10)
        sector_angle = 360/self.num_sectors
        for i in range(self.num_sectors):
            color = ['#FF0000', '#0000FF'][(i % 2)]
            ax.add_artist(Wedge((0, 0), 1, i * sector_angle, (i+1)*sector_angle, lw=0, color=color, alpha=0.1))
            ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.axis("off")
