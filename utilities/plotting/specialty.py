from scipy.cluster.hierarchy import dendrogram
from numpy import zeros, column_stack, ndarray
from matplotlib.pyplot import tight_layout, subplots, savefig, rcParams
from utilities.plotting.utilities import generate_3d_figure, set_labels
from config import font_size

rcParams.update({'font.size': font_size})


def plot_dendrogram(model, labels, title=None, filename=None, figsize=(10, 6)):
    """
    Plots the dendrogram of a sklearn agglomerative clustering model, based on code from the sklearn documentation

    :param model: sklearn agglomerative clustering model
    :param list labels: list of sample labels (that the model was trained on)
    :param str title: figure title
    :return: axis
    """

    # create the counts of samples under each node
    counts = zeros(model.children_.shape[0])
    num_samples = len(model.labels_)

    # Move through the tree and compute the linkage matrix
    for index, sub_tree in enumerate(model.children_):
        sub_tree_count = 0                                              # Reset current count
        for child_index in sub_tree:
            if child_index < num_samples:
                sub_tree_count += 1                                     # if the current node is a leaf node
            else:
                sub_tree_count += counts[child_index - num_samples]     # Get child count
        counts[index] = sub_tree_count

    linkage_matrix = column_stack([model.children_, model.distances_, counts]).astype(float)

    fig, ax = subplots(figsize=figsize)
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=80, ax=ax)

    if title is not None:
        ax.set_title(title)

    tight_layout()

    if filename is not None:
        savefig(filename)

    return ax


def plot_surface(x, y, z, title, filename=None, ax_labels=None, figsize=None, c_bar_title=None, azim=None, **args):
    """
    Plots a 3D surface

    :param ndarray x: Array of x values
    :param ndarray y: Array of y values
    :param ndarray z: Array of z values
    :param str title: Figure title
    :param Path filename: Path to save figure to [optional]
    :param tuple ax_labels: Tuple with axis titles
    :param tuple figsize: Custom figure size
    :param str c_bar_title: Colourbar title
    :param int azim: Default azimuth angle for 3d plot
    :return: Axis
    """

    # Generate figure
    fig, ax = generate_3d_figure(figsize=figsize)

    # Plot figure and set labels
    img = ax.plot_surface(x, y, z, **args)
    set_labels(ax, title, ax_labels)

    if 'cmap' in args:
        c_bar = fig.colorbar(img, ax=ax)
        if c_bar_title is not None:
            c_bar.ax.set_ylabel(c_bar_title)

    # Set default viewing angle
    if azim is not None:
        ax.azim = azim

    # Remove margin and save figure
    tight_layout()
    if filename is not None:
        savefig(filename)

    return ax
