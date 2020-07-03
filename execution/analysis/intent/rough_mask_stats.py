from utilities.data_management import make_path, load_vector, open_w_pandas, check_existence
from utilities.plotting import pie_chart, show
from numpy import sum, logical_not
from config import dataset

base = make_path('data/processed_data/') / dataset / 'analysis' / 'intent'
contexts_path = base / 'contexts.csv'
rough_mask_path = base / 'intent_mask.csv'
figure_path = make_path('figures') / dataset / 'analysis' / 'rough_mask_distribution.png'

check_existence([contexts_path, rough_mask_path])

contexts = open_w_pandas(contexts_path)
rough_mask = load_vector(rough_mask_path)

is_wikipedia = contexts['document_index'].values < 0
num_positive = sum(rough_mask == 1)
num_documents = rough_mask.shape[0]

print('Overall intent', num_positive / rough_mask.shape[0])
print('Intent in original dataset', num_positive / sum(logical_not(is_wikipedia)))
print('Unknown intent', sum(rough_mask == .5) / rough_mask.shape[0])

section_labels = ['positive', 'unknown', 'negative']

fractions = [
    sum(rough_mask == 1) / num_documents,
    sum(rough_mask == .5) / num_documents,
    sum(rough_mask == 0) / num_documents
]

pie_chart(fractions, section_labels, 'Rough mask class distribution', figure_path)
show()
