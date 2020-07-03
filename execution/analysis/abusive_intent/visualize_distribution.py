from utilities.data_management import make_path, load_vector
from utilities.plotting import show, hist_plot, plot_cumulative_distribution, plot_joint_distribution
from numpy import sum
import config

base = make_path('data/processed_data/') / config.dataset / 'analysis'
analysis_base = base / 'intent_abuse'

intent = load_vector(analysis_base / 'intent_predictions.csv')
abuse = load_vector(analysis_base / 'abuse_predictions.csv')
abusive_intent = load_vector(analysis_base / 'abusive_intent_predictions.csv')

midpoint = 0.4
print('Intent percentage', sum(intent > midpoint) / len(intent))
print('Intent percentage', sum(intent > midpoint))
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

show()
