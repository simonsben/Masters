from utilities.data_management import make_path, open_w_pandas, load_vector
from utilities.plotting import hist_plot, show, scatter_plot
import config

dataset = config.dataset

base = make_path('data/processed_data/') / dataset / 'analysis'
prediction_base = base / 'abusive_intent'
figure_base = make_path('figures/') / dataset / 'analysis'
hist_path = figure_base / 'num_context_histogram.png'

contexts = open_w_pandas(base / 'intent' / 'contexts.csv')

context_pieces = {}
amount_of_content = {}
for document_index, context in contexts[['document_index', 'contexts']].values:
    context_pieces[document_index] = 1 + (context_pieces[document_index] if document_index in context_pieces else 0)
    amount_of_content[document_index] = len(context) + (context_pieces[document_index] if document_index in context_pieces else 0)

num_contexts = list(context_pieces.values())
document_length = list(amount_of_content.values())

ax_titles = ('Number of contexts within document', 'Number of documents')

hist_plot(num_contexts, 'Histogram of number of contexts per document', hist_path, ax_titles)
scatter_plot((num_contexts, document_length), 'Number of contexts vs number of characters in document')

show()
