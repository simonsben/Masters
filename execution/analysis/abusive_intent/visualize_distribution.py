from utilities.data_management import make_path, load_vector, open_w_pandas, make_dir, get_prediction_path
from utilities.plotting import show, hist_plot, plot_cumulative_distribution, plot_joint_distribution, set_font_size
from model.analysis import estimate_joint_cumulative
from numpy import sum
from config import dataset

context_path = make_path('data/processed_data/') / dataset / 'analysis' / 'intent' / 'contexts.csv'
figure_base = make_path('figures') / dataset / 'analysis' / 'predictions'

make_dir(figure_base)

document_indexes = open_w_pandas(context_path)['document_index'].values
abuse = load_vector(get_prediction_path('abuse'))
intent = load_vector(get_prediction_path('intent'))
abusive_intent = load_vector(get_prediction_path('abusive_intent'))

not_wiki = document_indexes >= 0

midpoint = 0.5
print('Intent percentage', sum(intent > midpoint) / len(intent))
print('Storm-front intent percentage', sum(intent > midpoint) / sum(not_wiki))
print('Abuse percentage', sum(abuse > midpoint) / len(abuse))

ax_titles = ('Predicted abuse', 'Predicted intent')
hist_plot((abuse, intent), 'Joint histogram of abuse and intent', ax_titles=ax_titles, c_bar_title='Document density',
          filename=figure_base / 'joint_histogram.png')

set_font_size(16)

# Standard histograms
hist_plot(abuse, 'Histogram of abuse', figure_base / ('abuse-%s-hist.png' % dataset))
hist_plot(intent, 'Histogram of intent', figure_base / ('intent-%s-hist.png' % dataset))
hist_plot(abusive_intent, 'Histogram of abusive intent', figure_base / ('abusive_intent-%s-hist.png' % dataset))

# Cumulative normalized histogram
joint_distribution = estimate_joint_cumulative(abuse, intent)
normalized = joint_distribution(abuse, intent)
hist_plot(normalized, 'Histogram of normalized abusive intent',
          figure_base / 'normalized_abusive_intent-hist.png')


# Cumulative distributions
ax_labels = ('Predicted value', 'Cumulative sum')
plot_cumulative_distribution(abuse, 'Estimated cumulative distribution of abuse', ax_labels,
                             filename=figure_base / ('abuse-%s-cumulative.png' % dataset))
plot_cumulative_distribution(intent, 'Estimated cumulative distribution of intent', ax_labels,
                             filename=figure_base / ('intent-%s-cumulative.png' % dataset))

# Joint cumulative
set_font_size()
ax_labels = ('Intent prediction', 'Abuse prediction', 'Joint cumulative sum')
plot_joint_distribution(intent, abuse, 'Estimated joint abuse-intent cumulative distribution', ax_labels)

show()
