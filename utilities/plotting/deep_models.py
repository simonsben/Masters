from shap import DeepExplainer
from numpy import min, max, arange, sum, pi
from matplotlib.pyplot import subplots, savefig
from matplotlib import cm
from numpy.random import choice
from utilities.plotting.utilities import plot_sphere, plot_cone


def plot_token_importance(documents, indexed_documents, target_documents, model=None, path_generator=None, num_samples=10):
    """
    Generates a statistical representation of document token importance then plots it
    :param documents: list/array of (pre-processed) documents
    :param indexed_documents: list/array of documents with tokens replaced by embedding indexes
    :param target_documents: list/array of target document indexes (within indexed documents)
    :param model: compiled network model
    :param path_generator: function to return desired file path for figures
    :param num_samples: number of documents to use to model feature importance
    :return: None
    """

    # Get sample of documents to estimate feature importance with
    example_indexes = choice(indexed_documents.shape[0], num_samples, replace=False)
    example_documents = indexed_documents[example_indexes]

    explainer = DeepExplainer(model, example_documents)     # Generate statistical model for network

    # Compute feature importance for given documents
    shap_values = explainer.shap_values(indexed_documents[target_documents])
    print('shap value shape', shap_values.shape)
    shap_values = sum(shap_values, axis=2)
    # documents = documents[sample_inds]

    # for index, document, values in zip(range(num_samples), documents, shap_values):
    #     fig, ax = subplots()
    #
    #     words = document if type(document) == list else document.split(' ')     # Get list of words in document
    #     num_words = len(words)
    #
    #     values = values[:num_words].reshape(1, len(words))
    #     vmin, vmax = min(values), max(values)
    #
    #     img = ax.imshow(values, cmap=cm.Blues, vmin=vmin, vmax=vmax)
    #
    #     ax.set_xticks(arange(len(words)))
    #     ax.set_xticklabels(words, rotation=80)
    #
    #     fig.colorbar(img, ax=ax)
    #     ax.get_yaxis().set_visible(False)
    #
    #     if path_generator is not None:
    #         savefig(path_generator(index))


def plot_embedding_representation(target_norm, sphere_radius, cos_dist):
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



