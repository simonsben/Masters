from utilities.data_management import make_path, load_vector, open_w_pandas
from utilities.plotting import show, hist_plot, plot_cumulative_distribution, plot_joint_distribution
from numpy import sum
from config import dataset, mask_refinement_method

base = make_path('data/processed_data/') / dataset / 'analysis'
intent_base = base / 'intent'
analysis_base = base / 'intent_abuse'

refined = load_vector(intent_base / (mask_refinement_method + '_mask.csv'))
document_indexes = open_w_pandas(intent_base / 'contexts.csv')['document_index'].values
intent = load_vector(analysis_base / 'intent_predictions.csv')
abuse = load_vector(analysis_base / 'abuse_predictions.csv')
abusive_intent = load_vector(analysis_base / 'abusive_intent_predictions.csv')

not_wiki = document_indexes >= 0

midpoint = 0.5
print('Intent percentage', sum(intent > midpoint) / len(intent))
print('Storm-front intent percentage', sum(intent > midpoint) / sum(not_wiki))
print('Abuse percentage', sum(abuse > midpoint) / len(abuse))

ax_titles = ('Predicted abuse', 'Predicted intent')
hist_plot([abuse, intent], 'Prediction comparison histogram', ax_titles=ax_titles, c_bar_title='Document density')

hist_plot(abuse, 'Abuse histogram')
hist_plot(intent, 'Intent histogram')
hist_plot(abusive_intent, 'Abusive intent histogram')

ax_labels = ('Predicted value', 'Cumulative sum')

plot_cumulative_distribution(intent, 'Intent cumulative distribution', ax_labels)
plot_cumulative_distribution(abuse, 'Abuse cumulative distribution', ax_labels)

ax_labels = ('Intent prediction', 'Abuse prediction', 'Joint cumulative sum')
plot_joint_distribution(intent, abuse, 'Joint intent-abuse cumulative distribution', ax_labels)

# total = flip(
#     flip(cumsum(intent)) / intent.shape[0]
# )
# plot_line(total, 'Intent fraction based on cutoff')

# hist, edges = histogram(intent, bins=50)
# hist = hist / intent.shape[0]
#
# scatter_plot((edges[:-1], hist), 'bla')
# print(hist)

show()
