from matplotlib.pyplot import cm, subplots, title, savefig
from sklearn.metrics import confusion_matrix as calc_cm
from numpy import newaxis, sum, around


classes = ['neutral', 'abusive']


def confusion_matrix(predicted, labels, plot_title, filename=None):
    # Calculate confusion matrix and normalize
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
