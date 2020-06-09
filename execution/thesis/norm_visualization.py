from utilities.plotting import plot_surface, show
from utilities.data_management import make_path, make_dir
from numpy import linspace, meshgrid, abs, sqrt, max

# Define figure paths
figure_dir = make_path('figures/thesis/abusive_intent/')
make_dir(figure_dir)

one_path = figure_dir / 'one_norm.png'
two_path = figure_dir / 'two_norm.png'
infinity_path = figure_dir / 'infinity_norm.png'

# Generate meshed points
x = linspace(0, 1)
y = x.copy()
X, Y = meshgrid(x, y)

# Compute norms
Z_one = abs(X) + abs(Y)
Z_two = sqrt(X ** 2 + Y ** 2)
Z_infty = max([X, Y], axis=0)

# Normalize norms
Z_one /= max(Z_one)
Z_two /= max(Z_two)
Z_infty /= max(Z_infty)

# Define figure arguments
axis_labels = ('Abuse prediction', 'Intent prediction', 'Calculated abusive intent value')
colorbar_title = 'Norm value'
figure_size = (8, 5)
colormap = 'jet'
azim = 250

# Plot norm visualizations
plot_surface(X, Y, Z_one, 'Visualization of one-norm', one_path, axis_labels, figure_size, colorbar_title, azim,
             cmap=colormap, edgecolor='k', linewidth=.5)

plot_surface(X, Y, Z_two, 'Visualization of two-norm', two_path, axis_labels, figure_size, colorbar_title, azim,
             cmap=colormap, edgecolor='k', linewidth=.5)

plot_surface(X, Y, Z_infty, 'Visualization of infinity-norm', infinity_path, axis_labels, figure_size, colorbar_title,
             azim, cmap=colormap, edgecolor='k', linewidth=.5)

show()
