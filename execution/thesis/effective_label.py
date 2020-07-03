from numpy import linspace, asarray
from utilities.plotting import show, set_labels, tight_layout
from utilities.data_management import make_path, make_dir
from matplotlib.pyplot import subplots, savefig

figure_path = make_path('figures/thesis/validation/effective_weights.png')
make_dir(figure_path)


x = linspace(0, 1, 101)

y = x.copy()
y[x < .5] = -x[x < .5] + 1

values = asarray([x, y]).transpose()

fig, ax = subplots()
ax.plot(x, y)

ax.set_ylim(0, 1)
ax.set_xlim(0, 1)

axis_labels = ('Effective label', 'Sample weight')
set_labels(ax, 'Sample weighting based on effective label', axis_labels)

tight_layout()

savefig(figure_path)

show()
