from utilities.data_management import read_csv, move_to_root, make_path
from utilities.plotting import show, hist_plot, plot_cumulative_distribution, plot_joint_distribution
from utilities.analysis import rescale_data
from model.analysis import compute_abusive_intent
import config

move_to_root(4)
base = make_path('data/processed_data/') / config.dataset / 'analysis'
analysis_base = base / 'intent_abuse'

intent = read_csv(analysis_base / 'intent_predictions.csv', header=None)[0].values
abuse = read_csv(analysis_base / 'abuse_predictions.csv', header=None)[0].values

intent = rescale_data(intent)
abuse = rescale_data(abuse)

hybrid = compute_abusive_intent(intent, abuse)

ax_titles = ('Predicted abuse', 'Predicted intent')
hist_plot([abuse, intent], 'Prediction comparison histogram', ax_titles=ax_titles, c_bar_title='Document density')

hist_plot(abuse, 'Abuse histogram')
hist_plot(intent, 'Intent histogram')
hist_plot(hybrid, 'Abusive intent histogram')

ax_labels = ('Predicted value', 'Cumulative sum')

plot_cumulative_distribution(intent, 'Intent cumulative distribution', ax_labels)
plot_cumulative_distribution(abuse, 'Abuse cumulative distribution', ax_labels)

ax_labels = ('Intent prediction', 'Abuse prediction', 'Joint cumulative sum')
plot_joint_distribution(intent, abuse, 'Joint intent-abuse cumulative distribution', ax_labels)

show()
