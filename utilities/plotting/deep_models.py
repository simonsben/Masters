from shap import DeepExplainer
from numpy import min, max, arange, sum, pi
from matplotlib.pyplot import subplots, savefig
from matplotlib import cm
from numpy.random import randint
from utilities.plotting.utilities import plot_sphere, plot_cone


def word_importance(documents, embedded_docs, model=None, path_gen=None, num_samples=10):
    if model is None:
        shap_values = embedded_docs
    else:
        explainer = DeepExplainer(model, embedded_docs)
        sample_inds = randint(0, embedded_docs.shape[0], num_samples)
        [shap_values] = explainer.shap_values(embedded_docs[sample_inds])
        shap_values = sum(shap_values, axis=2)
        documents = documents[sample_inds]

    for ind, document, values in zip(range(num_samples), documents, shap_values):
        fig, ax = subplots()

        words = document if type(document) == list else document.split(' ')
        num_words = len(words)

        values = values[:num_words].reshape(1, len(words))
        vmin, vmax = min(values), max(values)

        img = ax.imshow(values, cmap=cm.Blues, vmin=vmin, vmax=vmax)

        ax.set_xticks(arange(len(words)))
        ax.set_xticklabels(words, rotation=80)

        fig.colorbar(img, ax=ax)
        ax.get_yaxis().set_visible(False)

        if path_gen is not None:
            savefig(path_gen(ind))


def plot_embedding_rep(target_norm, sphere_radius, cos_dist):
    """
    Plots a 3D representation of a higher dimensional set of vectors
    :param target_norm: Norm of the target vector
    :param sphere_radius: Radius of the sphere around the target vector
    :param cos_dist: Angle of the cone centered around the target vector
    :return: axis
    """
    # Define cone
    angle = cos_dist / 2 * pi
    ax = plot_cone((0, 0, 0), angle, target_norm)
    plot_sphere((0, 0, target_norm), sphere_radius, ax)

    return ax



