from utilities.data_management import read_csv, move_to_root, make_path, load_execution_params
from numpy import asarray, argsort, var, zeros_like, mean, log
from utilities.plotting import scatter_plot, show, hist_plot
from utilities.analysis import rescale_data

move_to_root(4)
params = load_execution_params()

base = make_path('data/processed_data/') / params['dataset'] / 'analysis'
analysis_base = base / 'intent_abuse'

intent = read_csv(analysis_base / 'intent_predictions.csv', header=None)[0].values
# abuse = read_csv(analysis_base / 'abuse_predictions.csv', header=None)[0].values
contexts = read_csv(base / 'intent' / 'contexts.csv')['contexts'].values
context_maps = read_csv(base / 'intent' / 'context_map.csv', names=['index', 'start', 'end'], index_col=0)
print('Content loaded.')

# Cap value range
intent = rescale_data(intent)
print('Content prepared.')


# context_slices = zeros_like(subset_mask, dtype=int)
# for doc_index, (start, end) in enumerate(context_maps.values):
#     for index in range(start, end + 1):
#         context_slices[index] = doc_index
# context_slices = context_slices[subset_mask]
print('Computed context slice mappings')

document_intents = {}
for intent_value, document_index in zip(intent, context_slices):
    if document_index in document_intents:
        document_intents[document_index].append(intent_value)
    else:
        document_intents[document_index] = [intent_value]
document_intents = list(document_intents[index] for index in sorted(document_intents))
print('Computed document intent lists')


means = zeros_like(document_intents, dtype=float)
variances = zeros_like(means)
lengths = asarray([len(values) for values in document_intents])
for index, intent_values in enumerate(document_intents):
    means[index] = mean(intent_values)
    variances[index] = var(intent_values)
print('Computed document-level intent variance and means')

means = means[lengths > 1]
variances = variances[lengths > 1]
lengths = lengths[lengths > 1]

# print(means[:20])
# print(variances[:20])

scatter_plot((means, variances), 'Intent vs document variance')
hist_plot(lengths, 'Context set lengths')
scatter_plot((log(lengths), means), 'Document contexts vs average intent')

sorted_variances = argsort(variances)

show()
