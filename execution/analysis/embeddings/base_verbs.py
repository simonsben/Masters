from fasttext import load_model
from utilities.data_management import load_execution_params, make_path, check_existence, move_to_root
from utilities.plotting import scatter_3_plot, show
from pandas import read_csv, DataFrame, read_hdf
from numpy import any, mean, std, log, asarray, percentile, sum, argsort
from os import remove
from scipy.linalg import svd
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

move_to_root(4)

params = load_execution_params()
dataset_name = params['dataset']
embedding_model = params['fast_text_model']

base_dir = make_path('data/processed_data') / dataset_name / 'analysis' / 'intent'
prediction_dir = base_dir.parent / 'intent_abuse'
embedding_path = make_path('data/lexicons/fast_text') / (embedding_model + '.bin')
verb_embeddings_path = base_dir / 'base_verb_embeddings.h5'
base_verb_path = base_dir / 'base_verbs.csv'
mask_path = base_dir / 'intent_mask.csv'
figure_path = make_path('figures/') / dataset_name / 'analysis' / 'base_verb_embeddings.png'

raw_base_verbs = read_csv(base_verb_path, header=None)[0].values
intent_predictions = read_csv(prediction_dir / 'intent_predictions.csv', header=None)[0].values
intent_mask = read_csv(mask_path, header=None)[0].values

good_bases = any([intent_mask == 0, intent_mask == 1, raw_base_verbs != 'None'], axis=0)
base_verbs = {}
base_predictions = {}
for verb, prediction in zip(raw_base_verbs[good_bases], intent_predictions[good_bases]):
    base_verbs[verb] = 1 + (base_verbs[verb] if verb in base_verbs else 0)

    if verb in base_predictions:
        base_predictions[verb].append(prediction)
    else:
        base_predictions[verb] = []
print('Number of unique base verbs', len(base_verbs))

embeddings = None
if verb_embeddings_path.exists():
    try:
        embeddings = read_hdf(verb_embeddings_path, key='embeddings')
    except (OSError, KeyError) as e:
        embeddings = None
        print('Bad file, re-computing embeddings')

if embeddings is None:
    model = load_model(str(embedding_path))
    print('Loaded fastText model')

    header = ['words', 'occurrences', 'intent_mean'] + [str(dimension) for dimension in range(model.get_dimension())]
    embeddings = DataFrame(
        [
            [verb, base_verbs[verb], mean(base_predictions[verb])] + list(model.get_word_vector(verb))
            for verb in base_verbs.keys()
         ], columns=header
    )
    embeddings.sort_values(by='occurrences', ascending=False, inplace=True)
    print('Generated embeddings')

    if verb_embeddings_path.exists():
        remove(verb_embeddings_path)
    embeddings.to_hdf(verb_embeddings_path, key='embeddings')

print('Embeddings loaded.')
print(embeddings)

embedding_vectors = embeddings.values[:, 3:].astype(float)

# Z-score embeddings
vector_mean = mean(embedding_vectors, axis=0)
vector_std = std(embedding_vectors, axis=0)
embedding_vectors = (embedding_vectors - vector_mean) / vector_std

occurrences = embeddings['occurrences'].values

percentile_value = 99
num_dimensions = 50
min_occurrences = 10

threshold = max(percentile(occurrences, percentile_value), min_occurrences)
high_occurrence = occurrences > threshold
print('Computed base verbs with threshold of', threshold)

vector_svds, _, _ = svd(embedding_vectors)
vector_svds = vector_svds[high_occurrence].transpose()[:num_dimensions]
print('Z-scored and computed svd of embeddings')

intent_means = embeddings['intent_mean'].values[high_occurrence]
correlations = [pearsonr(dimension, intent_means)[0] for dimension in vector_svds]
correlation_ordering = argsort(correlations)
target_dimensions = correlation_ordering[:3]

print('Plotting', sum(high_occurrence), 'embeddings')
axis_titles = ['SVD axis ' + str(index) for index in range(1, 4)]
scatter_3_plot(vector_svds[target_dimensions], str(percentile_value) + 'th percentile base verb embeddings',
               weights=intent_means, ax_titles=axis_titles,
               c_bar_title='Mean intent of documents', filename=figure_path, size=25)

pca = PCA()
single = pca.fit_transform(vector_svds.transpose())[:, 0]

verbs = asarray(list(base_verbs.keys()))
pairs = sorted(
    [(value, term) for value, term in zip(single, verbs[high_occurrence])],
    key=lambda bla: bla[0]
)

for pair in pairs:
    print(pair)

show()
