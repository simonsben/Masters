from dask.dataframe import read_csv
from dask.array.linalg import svd
from utilities.data_management import load_execution_params, make_path, move_to_root, check_existence
from matplotlib.pyplot import show, subplots

# Define paths
move_to_root(4)
embed_name = load_execution_params()['fast_text_model']
embed_path = make_path('data/prepared_lexicon/') / (embed_name + '.csv')
check_existence(embed_path)

# Define dataset-specific constants
dtypes = {str(ind): float for ind in range(1, 301)}
dtypes[0] = str


# Import data
embeddings = read_csv(embed_path, dtype=dtypes)
print('Data imported,', embeddings.shape[0].compute(), 'vectors')

vectors = embeddings.iloc[:, 1:]
u, s, v = svd(vectors.values)

s = s.compute()

fig, ax = subplots()
ax.scatter(range(len(s)), s, s=10)
ax.set_title('Singular values of word embeddings')
ax.set_xlabel('Feature dimension')
ax.set_ylabel('Singular value')
ax.set_yscale('log')

show()
