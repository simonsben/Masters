from matplotlib.pyplot import subplots, tight_layout, savefig, show
from numpy import arange, ndarray, asarray
from utilities.plotting.utilities import generate_3d_figure
from matplotlib.colors import LogNorm


def bar_plot(values, features, fig_title, filename=None, horizontal=False):
    """
    Generates a bar plot
    :param values: List of values
    :param features: List of features
    :param fig_title: Figure title
    :param filename: File path to save figure to, (default doesn't save)
    :param horizontal: Whether to use a horizontal bar plot, (default False)
    :return: (figure, axis) to allow further modification of plot
    """
    fig, ax = subplots()

    plot_type = ax.bar if not horizontal else ax.barh
    plot_type(list(range(len(values))), values)

    ax.set_xticks(arange(len(features)))
    ax.set_xticklabels(features, rotation='vertical')
    ax.set_title(fig_title)
    tight_layout()

    if filename is not None:
        savefig(filename)

    return fig, ax


def scatter_plot(values, fig_title, weights=None, filename=None, pre_split=True, c_bar_title=None, cmap='Blues', size=20):
    """
    Makes a 2D scatter plot
    :param values: List of values to be plotted
    :param fig_title: Title of figure
    :param weights: Weight of each point, (default uniform)
    :param filename: Filename for the figure, (default, doesn't save)
    :param pre_split: Whether the data is given in the shape 2,N or N,2, (default 2,N)
    :param c_bar_title: Title for the colourbar
    :param cmap: Colourmap color to use
    :return: Axis
    """
    fig, ax = subplots()

    # If not given as shape 2,N, split data, otherwise unpack
    if not pre_split:
        values = values if type(values) is ndarray else asarray(values)
        x, y = values[:, 0], values[:, 1]
    else:
        [x, y] = values

    img = ax.scatter(x, y, s=size, c=weights, cmap=cmap, edgecolors='k')

    # Add titles and colourbar (optional)
    ax.set_title(fig_title)
    if weights is not None:
        c_bar = fig.colorbar(img, ax=ax)
        if c_bar_title is not None:
            c_bar.ax.set_ylabel(c_bar_title)
    if filename is not None:
        savefig(filename)

    return ax


def scatter_3_plot(values, fig_title, weights=None, filename=None, ax_titles=None, cmap='Blues', c_bar_title=None, size=15):
    """
    Generates a 3D scatter plot
    :param values: Values to be plotted
    :param fig_title: Figure title
    :param weights: Weights of the data points, (default, uniform)
    :param filename: Filename for the figure
    :param ax_titles: Axis title
    :param cmap: Colourmap/colour scheme to use for weights
    :param c_bar_title: Colourbar title
    :return: Axis
    """
    fig, ax = generate_3d_figure()

    values = asarray(values) if type(values) is not ndarray else values
    if len(values) != 3:
        raise ValueError('Values should has shape (3, X), got', values.shape)
    if ax_titles is not None and len(ax_titles) != 3:
        raise ValueError('Axis titles should be a list or tuple with length 3')

    x, y, z = values
    if weights is None:
        ax.scatter(x, y, z)
    else:
        img = ax.scatter(x, y, z, c=weights, cmap=cmap, edgecolors='k', s=size)
        c_bar = fig.colorbar(img, ax=ax)
        if c_bar_title is not None:
            c_bar.ax.set_ylabel(c_bar_title)

    if ax_titles is not None:
        x_t, y_t, z_t = ax_titles
        ax.set_xlabel(x_t)
        ax.set_ylabel(y_t)
        ax.set_zlabel(z_t)
    ax.set_title(fig_title)

    if filename is not None:
        savefig(filename)

    return ax


def hist_plot(values, fig_title, filename=None, ax_titles=None, cmap='Blues', c_bar_title=None, bins=25, apply_log=True):
    """
    Generate a histogram of the provided data
    :param values: array of data or two arrays of data (i.e. shape (2, N))
    :param fig_title: title of figure
    :param filename: desired filename, optional
    :param ax_titles: axis titles, optional
    :param cmap: colourmap colour choice, optional
    :param c_bar_title: colourmap legend title, optional
    :param bins: number of bins, optional
    :param apply_log: apply log to data, optional
    :return: axis
    """

    fig, ax = subplots()
    values = asarray(values)    # Convert to numpy array

    if len(values.shape) > 1:   # If 2d data is provided, compute a 2d hist
        x, y = values
        norm = LogNorm() if apply_log else None
        img = ax.hist2d(x, y, bins=(bins, bins), cmap=cmap, norm=norm)[-1]

        c_bar = fig.colorbar(img, ax=ax)
        if c_bar_title is not None:
            c_bar.ax.set_ylabel(c_bar_title)
    else:                       # If single dimensional data is provided, compute a standard hist
        ax.hist(values, bins=bins, cmap=cmap, log=apply_log)

    if ax_titles is not None:   # If provided, add axis titles
        x_t, y_t = ax_titles
        ax.set_xlabel(x_t)
        ax.set_ylabel(y_t)
    ax.set_title(fig_title)     # Add title to figure
    tight_layout()              # Remove extra margin

    if filename is not None:    # If filename provided, save figure
        savefig(filename)

    return ax
