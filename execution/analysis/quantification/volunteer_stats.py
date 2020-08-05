from utilities.data_management import make_path, make_dir, open_w_pandas, check_existence
from utilities.plotting import hist_plot, show

label_path = make_path('data/datasets/data_labelling/labels.csv')
figure_path = make_path('figures/data_labelling/analysis/volunteer_responses.png')

check_existence(label_path)
make_dir(figure_path)

raw_labels = open_w_pandas(label_path)

print(len(raw_labels), 'total labels')

# Get number of unique respondents
num_volunteers = len(set(raw_labels['user_id']))
print(num_volunteers, 'unique volunteers')

# Get number of responses per respondent
votes_per_user = raw_labels.groupby('user_id').count()['context_id']
axis_labels = ('Number of labels submitted', 'Number of volunteers')
hist_plot(votes_per_user.values, 'Votes per volunteer', figure_path, ax_titles=axis_labels, apply_log=False, bins=5)

label_times = raw_labels['label_time']
print('Labelling took place between', label_times.min(), 'and', label_times.max())

show()
