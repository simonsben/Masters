from utilities.data_management import read_csv, move_to_root, make_path, load_execution_params, open_w_pandas
from utilities.plotting import show, hist_plot, plot_cumulative_distribution, plot_joint_distribution
from utilities.analysis import rescale_data
from model.analysis import compute_abusive_intent
from numpy import max, mean, zeros

move_to_root(4)
params = load_execution_params()

base = make_path('data/processed_data/') / params['dataset'] / 'analysis'
analysis_base = base / 'intent_abuse'

intent = read_csv(analysis_base / 'intent_predictions.csv', header=None)[0].values
abuse = read_csv(analysis_base / 'abuse_predictions.csv', header=None)[0].values
contexts = open_w_pandas(base / 'intent' / 'contexts.csv')
print('Loaded data')

intent = rescale_data(intent)
abuse = rescale_data(abuse)
# hybrid = compute_abusive_intent(intent, abuse)
print('Processed predictions')

document_abusive_intent = {}
for index, document_index in enumerate(contexts['document_index']):
    if document_index not in document_abusive_intent:
        document_abusive_intent[document_index] = []
    document_abusive_intent[document_index].append((abuse[index], intent[index]))

document_abusive_intent = list(document_abusive_intent.values())
print('Collected document abuse and intent')

num_documents = len(document_abusive_intent)
intent_mean = zeros(num_documents)
intent_max = zeros(num_documents)
abuse_mean = zeros(num_documents)
abuse_max = zeros(num_documents)

for index, values in enumerate(document_abusive_intent):
    means = mean(values, axis=0)
    maxes = max(values, axis=0)

    abuse_mean[index] = means[0]
    abuse_max[index] = maxes[0]

    intent_mean[index] = means[1]
    intent_max[index] = maxes[1]
print('Computed document max and mean values')

ax_titles = ('Predicted abuse', 'Predicted intent')

hist_plot([abuse_max, intent_max], 'Abuse vs intent - max, max', ax_titles=ax_titles)
hist_plot([abuse_max, intent_mean], 'Abuse vs intent - max, mean', ax_titles=ax_titles)
hist_plot([abuse_mean, intent_max], 'Abuse vs intent - mean, max', ax_titles=ax_titles)
hist_plot([abuse_mean, intent_mean], 'Abuse vs intent - mean, mean', ax_titles=ax_titles)

show()
