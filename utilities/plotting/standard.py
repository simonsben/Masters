from matplotlib.pyplot import subplots, tight_layout, savefig, show, rcParams
from numpy import arange, ndarray, asarray, min, max
from utilities.plotting.utilities import generate_3d_figure, set_labels
from matplotlib.colors import LogNorm
from pathlib import Path
from config import font_size

rcParams.update({'font.size': font_size})


def bar_plot(values, features, fig_title, filename=None, horizontal=False):
    """
    Generates a bar plot

    :param list values: List of values
    :param list features: List of features
    :param str fig_title: Figure title
    :param Path filename: File path to save figure to, (default doesn't save)
    :param bool horizontal: Whether to use a horizontal bar plot, (default False)
    :return: (figure, axis) to allow further modification of plot
    """
    fig, ax = subplots()

    plot_type = ax.bar if not horizontal else ax.barh
    plot_type(list(range(len(values))), values)

    ax.set_xticks(arange(len(features)))
    ax.set_xticklabels(features, rotation='vertical')
    set_labels(ax, fig_title, None)
    tight_layout()

    if filename is not None:
        savefig(filename)

    return fig, ax


def plot_line(values, fig_title, filename=None, ax_titles=None):
    """
    Makes a 2D line plot

    :param list values: List of values to be plotted
    :param str fig_title: Title of figure
    :param Path filename: Filename for the figure, (default, doesn't save)
    :param tuple ax_titles: Tuple containing the axis labels
    :return: Axis
    """
    fig, ax = subplots()
    ax.plot(values)

    set_labels(ax, fig_title, ax_titles)

    if filename is not None:
        savefig(filename)

    return ax


def scatter_plot(values, fig_title, weights=None, filename=None, ax_titles=None, pre_split=True, c_bar_title=None, cmap='Blues', size=20):
    """
    Makes a 2D scatter plot

    :param list values: List of values to be plotted
    :param str fig_title: Title of figure
    :param list weights: Weight of each point, (default uniform)
    :param Path filename: Filename for the figure, (default, doesn't save)
    :param bool pre_split: Whether the data is given in the shape 2,N or N,2, (default 2,N)
    :param str c_bar_title: Title for the colourbar
    :param str cmap: Colourmap color to use
    :return: Axis
    """
    fig, ax = subplots()

    # If not given as shape 2,N, split data, otherwise unpack
    if not pre_split:
        values = values if type(values) is ndarray else asarray(values)
        x, y = values[:, 0], values[:, 1]
    else:
        if len(values) == 2:
            [x, y] = values
        else:
            y = values
            x = arange(len(y))

    img = ax.scatter(x, y, s=size, c=weights, cmap=cmap, edgecolors='k')

    # Add titles and colourbar (optional)
    set_labels(ax, fig_title, ax_titles)

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

    :param list values: Values to be plotted
    :param str fig_title: Figure title
    :param list weights: Weights of the data points, (default, uniform)
    :param Path filename: Filename for the figure
    :param tuple ax_titles: Axis title
    :param str cmap: Colourmap/colour scheme to use for weights
    :param str c_bar_title: Colourbar title
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

    set_labels(ax, fig_title, ax_titles)

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
        ax.hist(values, bins=bins, log=apply_log)
        ax.autoscale(enable=True, axis='x', tight=True)

    set_labels(ax, fig_title, ax_titles)
    tight_layout()              # Remove extra margin

    if filename is not None:    # If filename provided, save figure
        savefig(filename)

    return ax


def pie_chart(values, labels, figure_title, filename=None):
    """
    Generates a pie chart

    :param list values: List of category fractions
    :param list labels: List of labels for each of the category fractions
    :param str figure_title: Title for the figure
    :param Path filename: File path to save the figure to, optional
    """
    fig, ax = subplots()

    ax.pie(values, labels=labels, autopct='%1.1f%%', shadow=True)

    ax.set_title(figure_title)
    tight_layout()

    if filename is not None:    # If filename provided, save figure
        savefig(filename)

    return ax


def stacked_plot(x_values, y_values, labels, figure_title, axis_labels=None, filename=None, figsize=None):
    """
    Generates a stacked plot (stacked area plot)

    :param ndarray x_values: Array of N x-values
    :param ndarray y_values: Array of MxN y-values
    :param list labels: List of labels for each of the M categories
    :param str figure_title: Figure title
    :param tuple[str, str] axis_labels: X and Y axis labels, optional
    :param Path filename: Path to save generated figure to, optional
    :param tuple[int, int] figsize: Size of the generated figure, optional
    :return: Axis
    """
    fig, ax = subplots(figsize=figsize)
    ax.stackplot(x_values, y_values, labels=labels, edgecolor='k', linewidth=1)

    ax.set_xlim(min(x_values), max(x_values))
    ax.set_ylim(0, 1)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower right')

    set_labels(ax, figure_title, axis_labels)

    tight_layout()  # Remove extra margin

    if filename is not None:  # If filename provided, save figure
        savefig(filename)

    return ax
