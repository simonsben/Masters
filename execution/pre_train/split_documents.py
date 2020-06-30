from utilities.data_management import make_path, make_dir, open_w_pandas, check_existence, save_dataframe
from model.extraction import split_into_contexts
from pandas import DataFrame
from utilities.pre_processing import final_clean
from config import dataset

data_path = make_path('data/prepared_data') / (dataset + '_partial.csv')
context_path = make_path('data/processed_data') / dataset / 'analysis' / 'intent' / 'contexts.csv'

check_existence([data_path])
make_dir(context_path)

raw_documents = open_w_pandas(data_path)
documents = raw_documents['document_content'].values

document_contexts, (document_indexes, context_indexes) = split_into_contexts(documents, raw_documents.index.values)
print('Contexts extracted, expanded', len(documents), 'to', len(document_contexts))

document_contexts = DataFrame(list(map(final_clean, document_contexts)), columns=['contexts'])
document_contexts['document_index'] = document_indexes
document_contexts['context_index'] = context_indexes
print('Assembled processed data.')

# Save contexts
save_dataframe(document_contexts, context_path)
print('Contexts saved.')

