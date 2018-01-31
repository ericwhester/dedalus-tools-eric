import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize


cdict_K_black = { 'red': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
           'green':((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
           'blue': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
           'alpha':((0.0, 0.0, 0.0), (1.0, 0.05, 0.05)) }
cmap_K = LinearSegmentedColormap('cmap_K', cdict_K_black)
plt.register_cmap(cmap=cmap_K)

def color_norm(vmin, vmax):
    """Wrapper to create a normalize object to scale a colorbar (particularly for quiver plots."""
    return Normalize(vmin=vmin, vmax=vmax)
