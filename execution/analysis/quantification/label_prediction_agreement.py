from utilities.data_management import make_path, load_vector, check_existence, open_w_pandas
from utilities.plotting import hist_plot, show
from krippendorff import alpha as compute_alpha
from sklearn.metrics import cohen_kappa_score
from numpy import vstack, any, where

base = make_path('data/processed_data/data_labelling/analysis/intent_abuse')
label_path = base / 'labels.csv'
prediction_path = base / 'intent_predictions.csv'

check_existence([label_path, prediction_path])

raw_labels = open_w_pandas(label_path)
predictions = load_vector(prediction_path)
print('Loaded data.')

[enough_labels] = where(any(raw_labels[['0', '1']].values >= 3, axis=1))
labels = raw_labels['rating'].values[enough_labels]

reliability_data = vstack([labels, predictions[enough_labels]])
print('Constructed reliability data')

alpha = compute_alpha(reliability_data)
cat_alpha = compute_alpha(reliability_data >= .5, level_of_measurement='ordinal')

kappa = cohen_kappa_score(labels > .5, predictions[enough_labels] > .5)

print('Alpha: %.3f' % alpha)
print('Alpha with rounded values: %.3f' % cat_alpha)
print('Kappa: %.3f' % kappa)

hist_plot(labels - predictions[enough_labels], 'Deltas', apply_log=False)

show()
