from utilities.data_management import read_csv, move_to_root, make_path
from numpy import argsort, log2, sqrt, power
from utilities.plotting import scatter_plot, show, hist_plot
from utilities.analysis import rescale_data, list_means, list_variances, list_lengths, list_mins, list_maxes
from pandas import DataFrame
from scipy.spatial.distance import euclidean
import config
move_to_root(4)
base = make_path('data/processed_data/') / config.dataset / 'analysis'
analysis_base = base / 'intent_abuse'

intent = read_csv(analysis_base / 'intent_predictions.csv', header=None)[0].values
# abuse = read_csv(analysis_base / 'abuse_predictions.csv', header=None)[0].values
raw_contexts = read_csv(base / 'intent' / 'contexts.csv')
contexts = raw_contexts['contexts'].values

context_maps = read_csv(base / 'intent' / 'context_map.csv', names=['index', 'start', 'end'], index_col=0)
print('Content loaded.')

# Cap value range
intent = rescale_data(intent)
print('Content prepared.')

document_intentions = [intent[start:end+1] if end >= start else [0] for start, end in context_maps.values]

statistics = DataFrame(list_means(document_intentions), columns=['mean'])
statistics['min'] = list_mins(document_intentions)
statistics['max'] = list_maxes(document_intentions)
statistics['variance'] = list_variances(document_intentions)
statistics['length'] = list_lengths(document_intentions)

min_contexts = 2
# statistics.sort_values('length', inplace=True, ascending=False)
long_documents = statistics['length'] >= min_contexts
# statistics = statistics.loc[long_documents]

means, mins, maxes, variances, lengths = statistics.values.transpose()
log_lengths = log2(lengths)
print('Statistics computed.')

c_bar_title = 'Logged number of document contexts'
y_label = 'Variance of document intent'

# scatter_plot((means, variances), 'Intent vs document variance', log_lengths,
#              ax_titles=('Average document intent', y_label), c_bar_title=c_bar_title)
#
# scatter_plot((means, maxes), 'means vs maxes', log_lengths,
#              ax_titles=('mean intent', 'max intent'), c_bar_title='intent variance')
# scatter_plot((log_lengths, variances), 'lengths vs variances', means)

# hist_plot(lengths, 'Context set lengths')
# scatter_plot((log_lengths, means), 'Document contexts vs average intent',
#              ax_titles=('Logged number of document contexts', 'Average document intent'))

hybrid = sqrt(power(means, 2) + power(maxes, 2))
hybrid_order = argsort(hybrid)
hybrid_order = hybrid_order[lengths[hybrid_order] > 1]

for document_index in reversed(hybrid_order[-40:]):
    tmp = context_maps.iloc[document_index]
    print(document_index, hybrid[document_index], means[document_index], maxes[document_index])
    for context_index in range(tmp[0], tmp[1]+1):
        print('\t', context_index, intent[context_index], contexts[context_index])

show()
