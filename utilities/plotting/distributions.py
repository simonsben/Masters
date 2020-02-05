from matplotlib.pyplot import subplots, tight_layout
from numpy import ndarray, linspace, vectorize, arange, meshgrid
from model.analysis import estimate_cumulative, estimate_joint_cumulative
from utilities.plotting.utilities import generate_3d_figure, set_labels


def plot_cumulative_distribution(data, fig_title, ax_labels=None, resolution=.01):
    """
    Plots the cumulative distribution of a dataset
    :param data: Dataset vector, numpy array
    :param fig_title: Figure title, string
    :param ax_labels: Axis labels, tuple
    :param resolution: Resolution of distribution estimation, float
    :return: axis
    """
    if not isinstance(data, ndarray):
        return TypeError('Expected data as numpy array.')

    cumulative_function = estimate_cumulative(data, num_bins=int(1 / resolution * 2))
    cumulative_function = vectorize(cumulative_function)
    reference = linspace(0, 1, 50)
    cumulative = cumulative_function(reference)

    fig, ax = subplots()
    ax.plot(reference, cumulative, '.-')

    ax.set_title(fig_title)

    set_labels(ax, fig_title, ax_labels)
    tight_layout()

    return ax


def plot_joint_distribution(data_a, data_b, fig_title, ax_labels=None, resolution=.01):
    """
    Plots the joint (but assumed independent) distribution of two sets of data
    :param data_a: First dataset vector, numpy array
    :param data_b: Second dataset vector, numpy array
    :param fig_title: Figure title, string
    :param ax_labels: Axis labels, tuple of stings
    :param resolution: Resolution of distribution estimation, float
    :return: axis
    """
    if not isinstance(data_a, ndarray) or not isinstance(data_b, ndarray):
        return TypeError('Expected data as numpy array.')

    joint_cumulative_function = estimate_joint_cumulative(data_a, data_b, resolution)

    x, y = [arange(0, 1, resolution) for _ in range(2)]
    x, y = meshgrid(x, y)

    z = joint_cumulative_function(x, y)

    fig, ax = generate_3d_figure()
    ax.plot_surface(x, y, z)

    set_labels(ax, fig_title, ax_labels)
    tight_layout()

    return ax
