from scipy.cluster.hierarchy import dendrogram
from numpy import zeros, column_stack
from matplotlib.pyplot import tight_layout, subplots, savefig


def plot_dendrogram(model, labels, title=None, filename=None, figsize=(10, 6)):
    """
    Plots the dendrogram of a sklearn agglomerative clustering model, based on code from the sklearn documentation
    :param model: sklearn agglomerative clustering model
    :param labels: list of sample labels (that the model was trained on)
    :param title: figure title
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
