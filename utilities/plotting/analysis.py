from matplotlib.pyplot import cm, subplots, title, savefig, tight_layout
from sklearn.metrics import confusion_matrix as calc_cm
from numpy import newaxis, sum, around, arange, array, linspace, abs
from shap import TreeExplainer


classes = ['neutral', 'abusive']


def confusion_matrix(predicted, labels, plot_title, filename=None):
    # Calculate confusion matrix and normalize
    if 'int' not in str(predicted.dtype):
        predicted = around(predicted).astype(int)

    confusion = calc_cm(labels, predicted)
    confusion = around(confusion / sum(confusion, axis=1)[:, newaxis] * 100, decimals=1)

    # Create figure and plot
    fig, ax = subplots()
    im = ax.imshow(confusion, cmap=cm.Greens, vmin=0, vmax=100)
    ax.figure.colorbar(im, ax=ax)

    # Correct tick scale and add labels
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('True Value')

    # Add values to confusion matrix
    for p_ind, row in enumerate(confusion):     # For each predicted-value index
        for t_ind, val in enumerate(row):       # For each true-value index
            ax.text(t_ind, p_ind, val, ha='center', va='center',
                    # If square is dark, use white, else black for text
                    color='w' if val > 50 else 'k')

    title(plot_title)

    if filename is not None:
        savefig(filename)


def feature_significance(feature_weights, figure_title, filename=None, max_features=20, is_weight=True, x_log=False):
    feature_weights = array(feature_weights[:max_features])

    y_ticks = arange(len(feature_weights))
    x_ticks = around(linspace(0, max(feature_weights[:, 1].astype(float)), 10, dtype=float), decimals=1)

    fig, ax = subplots()
    ax.barh(y_ticks, feature_weights[:, 1].astype(float), edgecolor='k')

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(feature_weights[:, 0])
    ax.set_xticklabels(x_ticks)
    ax.set_ylabel('Feature')
    ax.set_xlabel('Weight' if is_weight else 'Gain')

    if x_log:
        ax.set_xscale('log')

    title(figure_title)
    fig.tight_layout()

    if filename is not None:
        savefig(filename)


def shap_feature_significance(model, dataset, figure_title, features=None, filename=None):
    """ Generates a bar plot of the SHAP feature weights """
    shap_values = abs(
        TreeExplainer(model).shap_values(dataset)
    ).mean(0)

    features = dataset.columns.values if features is None else features
    feature_shaps = sorted(
        [(features[ind], shap_value) for ind, shap_value in enumerate(shap_values)],
        key=lambda shap: shap[1],
        reverse=True
    )
    feature_significance(feature_shaps, figure_title, filename, x_log=True)


def bar_plot(values, features, fig_title, filename=None):
    """ Generates a bar plot """
    fig, ax = subplots()

    ax.bar(list(range(len(values))), values)

    ax.set_xticks(arange(len(features)))
    ax.set_xticklabels(features, rotation='vertical')
    title(fig_title)
    tight_layout()

    if filename is not None:
        savefig(filename)

    return ax
