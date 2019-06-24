from shap import DeepExplainer
from numpy import min, max, arange, array
from matplotlib.pyplot import subplots, savefig
from matplotlib import cm
from numpy.random import randint


def word_importance(documents, embedded_docs, model=None, path_gen=None, num_samples=10):
    if model is None:
        shap_values = embedded_docs
    else:
        explainer = DeepExplainer(model, embedded_docs)
        sample_ind = randint(0, embedded_docs.shape[0], num_samples)
        shap_values = [explainer.shap_values(document)[0] for document in embedded_docs[sample_ind]]

    for ind, document, values in zip(range(len(documents)), documents, shap_values):
        fig, ax = subplots()

        words = document if type(document) == list else document.split(' ')
        values = values[:len(words)]
        vmin, vmax = min(values), max(values)

        img = ax.imshow(values, cmap=cm.Blues, vmin=vmin, vmax=vmax)

        ax.set_xticks(arange(len(words)))
        ax.set_xticklabels(words, rotation=45)

        ax.figure.colorbar(img, ax=ax)
        ax.get_yaxis().set_visible(False)

        if path_gen is not None:
            savefig(path_gen(ind))

# features = [('Dimension ' + str(i)) for i in range(len(shap_values))]
# feature_weights = sorted(
#     [(feature, weight) for feature, weight in zip(features, shap_values)],
#     reverse=True, key=lambda doc: doc[1]
# )
#
# # Deep model
# feature_significance(feature_weights, 'FastText BiLSTM SHAP Weights',
#                      filename=shap_dir / 'deep_weight.png')