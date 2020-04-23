from utilities.data_management import move_to_root, make_path, open_w_pandas, load_vector
from utilities.plotting import hist_plot, show, scatter_plot
import config

move_to_root()

dataset = config.dataset

base = make_path('data/processed_data/') / dataset / 'analysis'
prediction_base = base / 'abusive_intent'

contexts = open_w_pandas(base / 'intent' / 'contexts.csv')

context_pieces = {}
amount_of_content = {}
for document_index, context in contexts[['document_index', 'contexts']].values:
    context_pieces[document_index] = 1 + (context_pieces[document_index] if document_index in context_pieces else 0)
    amount_of_content[document_index] = len(context) + (context_pieces[document_index] if document_index in context_pieces else 0)

num_contexts = list(context_pieces.values())
document_length = list(amount_of_content.values())

hist_plot(num_contexts, 'Histogram of number of contexts per document')
scatter_plot((num_contexts, document_length), 'Number of contexts vs number of characters in document')

show()
