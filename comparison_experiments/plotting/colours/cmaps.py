import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

cmaps = ["Greens", "Oranges", "Blues", "Greys", "Reds", "Purples", "YlOrBr"]
palettes = [
    sns.color_palette(cmap, n_colors=8)[3:] for cmap in cmaps
]  # exclude first few colours to avoid white
palette_cmaps = [
    LinearSegmentedColormap.from_list(name="cmap", colors=p, N=len(p)) for p in palettes
]
