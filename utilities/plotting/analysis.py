from matplotlib.pyplot import cm, subplots, title, savefig, Axes
from sklearn.metrics import confusion_matrix as calc_cm
from numpy import newaxis, sum, around, arange, array, linspace, abs, size, where, max
from shap import TreeExplainer
from utilities.plotting.utilities import set_labels


classes = ['negative', 'positive']
axis_labels = ('Predicted Value', 'Label Value')

def confusion_matrix(predicted, labels, plot_title, filename=None, threshold=.5, ax_labels=axis_labels):
    """
    Generates confusion matrix for a given set of predicted and labelled data
    :param predicted: List of predictions
    :param labels: Labels corresponding to predicted data
    :param plot_title: Title of figure
    :param filename: Filename to save figure to, (default doesn't save)
    :param threshold: Threshold value for prediction rounding, (default .5)
    :return:
    """
    if size(predicted) != size(labels):
        raise ValueError('Predicted values and labels must be of the same length')

    # Calculate confusion matrix and round
    if 'int' not in str(predicted.dtype):
        predicted = where(predicted > threshold, 1, 0).astype(int)

    confusion = calc_cm(labels, predicted)
    confusion = around(confusion / sum(confusion, axis=1)[:, newaxis] * 100, decimals=1)

    # Create figure and plot
    fig, ax = subplots()
    ax.imshow(confusion, cmap=cm.Greens, vmin=0, vmax=100)

    # Correct tick scale and add labels
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Add values to confusion matrix
    for p_ind, row in enumerate(confusion):     # For each predicted-value index
        for t_ind, val in enumerate(row):       # For each true-value index
            ax.text(t_ind, p_ind, ('%.1f%%' % val), ha='center', va='center',
                    # If square is dark, use white, else black for text
                    color=('w' if val > 50 else 'k'))

    set_labels(ax, plot_title, ax_labels)

    if filename is not None:
        savefig(filename)


def feature_significance(feature_weights, figure_title, filename=None, max_features=50, x_log=False, figsize=(8, 8)):
    """
    Generates a bar plot of feature significance values
    :param list feature_weights: List of tuples in the form (feature name, feature weight)
    :param str figure_title: Figure title
    :param Path filename: File path to save figure to, (default doesn't save)
    :param int max_features: Maximum number of features to include in plot (default 20)
    :param bool x_log: Whether the values should be plotted on a log scale, (default False)
    :return Axes: Figure axis
    """
    tokens, weights = array(feature_weights[:max_features]).T
    weights = weights.astype(float)

    y_ticks = arange(len(tokens))
    x_ticks = around(linspace(0, max(weights), 10), decimals=1)

    fig, ax = subplots(figsize=figsize)
    ax.barh(y_ticks, weights, edgecolor='k')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(tokens)

    ax.set_ylabel('Token')
    ax.set_xlabel('SHAP Value')

    if x_log:
        ax.set_xscale('log')

    title(figure_title)
    fig.tight_layout()

    if filename is not None:
        savefig(filename)

    return ax


def shap_feature_significance(model, document_matrix, figure_title, features=None, filename=None):
    """
    Generates a bar plot of the SHAP feature weights
    :param model: Trained XGBoost model
    :param document_matrix: Pandas SparseDataFrame with predicted values -> (documents x features)
    :param figure_title: Figure title
    :param features: Feature names, (default dataset column names)
    :param filename: File path to save figure to, (default doesn't save)
    """
    shap_values = abs(
        TreeExplainer(model).shap_values(document_matrix)
    ).mean(0)

    features = document_matrix.columns.values if features is None else features
    feature_shaps = sorted(
        [(features[ind], shap_value) for ind, shap_value in enumerate(shap_values)],
        key=lambda shap: shap[1],
        reverse=True
    )
    feature_significance(feature_shaps, figure_title, filename, x_log=True)
