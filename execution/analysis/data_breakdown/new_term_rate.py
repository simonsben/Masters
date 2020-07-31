from utilities.data_management import make_dir, make_path, open_w_pandas, check_existence, vector_to_file, load_vector
from utilities.plotting import plot_line, show
from numpy import zeros_like, cumsum
from config import dataset

computed_path = make_path('data/processed_data') / 'thesis' / 'data' / 'unique_term_rate.csv'
data_path = make_path('data/prepared_data') / (dataset + '.csv')
figure_path = make_path('figures') / dataset / 'analysis' / 'new_term_rate.png'

check_existence(data_path)
make_dir(figure_path)
make_dir(computed_path)
print('Config complete')

if not computed_path.exists():
    documents = open_w_pandas(data_path)['document_content'].values
    term_counts = zeros_like(documents, dtype=int)
    print('Loaded data')

    unique_terms = set()
    for index, document in enumerate(documents):
        if not isinstance(document, str):
            continue

        for token in document.split(' '):
            if token not in unique_terms:
                unique_terms.add(token)
                term_counts[index] += 1

    unique_term_coverage = cumsum(term_counts) / len(unique_terms)
    print('Computed new term rate')

    vector_to_file(unique_term_coverage, computed_path)
    print('Saved computed data')
else:
    unique_term_coverage = load_vector(computed_path)

axis_labels = ('Number of documents', 'Fraction of unique terms')
plot_line(unique_term_coverage, 'Cumulative distribution of unique tokens', figure_path)
print('Complete.')

show()
