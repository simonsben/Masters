from fastText import load_model
from tables.exceptions import HDF5ExtError
from utilities.data_management import load_execution_params, make_path, check_existence, move_to_root
from utilities.plotting import scatter_3_plot, show
from pandas import read_csv, DataFrame, read_hdf
from numpy import any, mean, std, log, asarray, percentile, sum
from os import remove
from scipy.linalg import svd
from sklearn.decomposition import PCA

move_to_root(4)

params = load_execution_params()
dataset_name = params['dataset']
embedding_model = params['fast_text_model']

base_dir = make_path('data/processed_data') / dataset_name / 'analysis' / 'intent'
embedding_path = make_path('data/lexicons/fast_text') / (embedding_model + '.bin')
verb_embeddings_path = base_dir / 'base_verb_embeddings.h5'
base_verb_path = base_dir / 'base_verbs.csv'
mask_path = base_dir / 'intent_mask.csv'
figure_path = make_path('figures/') / dataset_name / 'analysis' / 'base_verb_embeddings.png'

raw_base_verbs = read_csv(base_verb_path, header=None)[0].values
intent_mask = read_csv(mask_path, header=None)[0].values

good_bases = any([intent_mask == 0, intent_mask == 1, raw_base_verbs != 'None'], axis=0)
base_verbs = {}
for verb in raw_base_verbs[good_bases]:
    base_verbs[verb] = 1 + (base_verbs[verb] if verb in base_verbs else 0)
print('Number of unique base verbs', len(base_verbs))

embeddings = None
if verb_embeddings_path.exists():
    try:
        embeddings = read_hdf(verb_embeddings_path, key='embeddings')
    except (OSError, HDF5ExtError, KeyError) as e:
        embeddings = None
        print('Bad file, re-computing embeddings')

if embeddings is None:
    model = load_model(str(embedding_path))
    print('Loaded fastText model')

    embeddings = DataFrame(
        [[verb, base_verbs[verb]] + list(model.get_word_vector(verb)) for verb in base_verbs.keys()],
        columns=(['words', 'occurrences'] + [str(dimension) for dimension in range(model.get_dimension())])
    )
    embeddings.sort_values(by='occurrences', ascending=False, inplace=True)
    print('Generated embeddings')

    if verb_embeddings_path.exists():
        remove(verb_embeddings_path)
    embeddings.to_hdf(verb_embeddings_path, key='embeddings')

print('Embeddings loaded.')
print(embeddings)

embedding_vectors = embeddings.values[:, 2:].astype(float)

# Z-score embeddings
vector_mean = mean(embedding_vectors, axis=0)
vector_std = std(embedding_vectors, axis=0)
embedding_vectors = (embedding_vectors - vector_mean) / vector_std

occurrences = asarray([base_verbs[verb] for verb in base_verbs])
percentile_value = 99.6
threshold = max(percentile(occurrences, percentile_value), 10)
high_occurrence = occurrences > threshold
print('Computed base verbs with threshold of', threshold)

vector_svds, _, _ = svd(embedding_vectors)
vector_svds = vector_svds[high_occurrence][:, :3].transpose()
print('Z-scored and computed svd of embeddings')

print('Plotting', sum(high_occurrence), 'embeddings')
axis_titles = ['SVD axis ' + str(index) for index in range(1, 4)]
scatter_3_plot(vector_svds, str(percentile_value) + 'th percentile base verb embeddings',
               weights=log(occurrences[high_occurrence]), ax_titles=axis_titles,
               c_bar_title='Log of number of occurrences', filename=figure_path)

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
