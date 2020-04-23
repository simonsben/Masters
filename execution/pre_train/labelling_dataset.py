from utilities.data_management import read_csv, make_path, open_w_pandas
from numpy import argsort, sum, any, where, zeros, all, asarray
from numpy.random import choice
from utilities.analysis import rescale_data
from model.analysis import compute_abusive_intent, estimate_cumulative
from utilities.plotting import hist_plot, show
import config

dataset = config.dataset

base = make_path('data/processed_data/') / dataset / 'analysis'
analysis_base = base / 'intent_abuse'
intent_base = base / 'intent'
figure_base = make_path('figures/') / dataset / 'analysis'

english_mask = read_csv(intent_base / 'english_mask.csv', header=None)[0].values.astype(bool)
intent = read_csv(analysis_base / 'intent_predictions.csv', header=None)[0].values[english_mask]
abuse = read_csv(analysis_base / 'abuse_predictions.csv', header=None)[0].values[english_mask]

contexts = open_w_pandas(intent_base / 'contexts.csv').loc[english_mask]
# contexts = raw_contexts['contexts'].values
print('Content loaded.')

# Rescale value range
abuse = rescale_data(abuse)
intent = rescale_data(intent)
print('Content prepared.')

# Remove wikipedia contexts used for training
non_wikipedia = contexts['document_index'].values >= 0
contexts, intent, abuse = contexts.loc[non_wikipedia], intent[non_wikipedia], abuse[non_wikipedia]

# Compute the euclidean norm of the (abuse, intent) vectors for each context
abusive_intent = compute_abusive_intent(intent, abuse)
print('Finished computations.')

limit = .40
zone_width = .025
total_samples = 5000

num_zones = int(limit * 1.5 / zone_width)
samples_per_zone = int(total_samples / num_zones)

zone_indexes = zeros((num_zones, samples_per_zone), dtype=int)
zone_indexes[:] = -1
for zone_number in range(num_zones):
    base = round(
        zone_number * zone_width + ((1 - limit * 1.5) if zone_number >= num_zones / 3 else 0),
        3
    )
    top = round(base + zone_width, 3)
    [indexes] = where(all([abusive_intent > base, abusive_intent <= top], axis=0))

    print(base, top, 'zone size', indexes.shape[0])
    chosen = choice(indexes.shape[0], min(samples_per_zone, indexes.shape[0]), replace=False)

    sample = indexes[chosen]
    zone_indexes[zone_number, :sample.shape[0]] = sample

print('samples per zone', samples_per_zone)

zone_indexes = zone_indexes.flatten()
zone_indexes = zone_indexes[zone_indexes != -1]

num_indexes = zone_indexes.shape[0]
zone_indexes = zone_indexes[choice(num_indexes, num_indexes, replace=False)]

selected_contexts = contexts.iloc[zone_indexes]
selected_contexts = selected_contexts.loc[selected_contexts['contexts'].values != ' ']
selected_contexts.to_csv(analysis_base / 'labelling_contexts.csv', index_label='context_id')

hist_plot(abusive_intent, 'Abusive intent histogram', figure_base / 'abusive_intent_histogram.png')
hist_plot(abusive_intent[zone_indexes], 'Selected abusive intent histogram', figure_base / 'selected_abusive_intent_histogram.png')

show()
