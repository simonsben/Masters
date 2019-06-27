from shap import DeepExplainer
from numpy import min, max, arange, sum, linspace, meshgrid, pi, outer, sin, cos, arctan, ones_like
from matplotlib.pyplot import subplots, savefig, figure
from matplotlib import cm
from numpy.random import randint
from mpl_toolkits.mplot3d import Axes3D


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
    radius = linspace(0, 35)
    polar_angles = linspace(0, 2 * pi)
    radius_points, polar_points = meshgrid(radius, polar_angles)

    # Convert to cartesian
    Z = radius_points / arctan(cos_dist / 2 * pi)
    X, Y = radius_points * cos(polar_points), radius_points * sin(polar_points)

    # Plot cone
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=.7)

    # Define sphere
    u = linspace(0, pi)

    # Convert to cartesian
    X = sphere_radius * outer(sin(u), sin(polar_angles))
    Y = sphere_radius * outer(sin(u), cos(polar_angles))
    Z = (sphere_radius * outer(cos(u), ones_like(polar_angles))) + target_norm

    # Plot sphere
    ax.plot_surface(X, Y, Z, color='g', alpha=.7)

    return ax
