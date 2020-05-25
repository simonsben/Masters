from matplotlib.pyplot import subplots, show, savefig
from matplotlib import rcParams
from numpy import sin, cos, arctan, asarray, min, max, sqrt, linspace
from utilities.data_management import make_dir, make_path

rcParams.update({'font.size': 14})

# Define set of example points
example_points = asarray([
    (1, 3),
    (1.3, 2),
    (1.5, 2.5),
    (1.6, 2.8),
    (1.2, 2.2),
    (1.3, 2.9),
    (1.5, 2),
    (1.7, 2.4),
    (1.9, 2.7),
    (1.1, 2.7)
])

# Compute slopes and angles
slopes = [y / x for x, y in example_points]
angles = arctan(slopes)

# Compute point magnitudes
x, y = example_points.transpose()
radius = sqrt(x ** 2 + y ** 2)

# Plot points
fig, ax = subplots()
ax.scatter(x, y)

# Get min and max x and y values
min_x, min_y = min(x), min(y)
max_x, max_y = max(x), max(y)

# Plot box constraint
ax.plot((min_x, min_x, max_x, max_x, min_x), (min_y, max_y, max_y, min_y, min_y), 'g')

# Get min and max angle and radius values
min_r, max_r = min(radius), max(radius)
min_a, max_a = min(angles), max(angles)

# Generate angle array
theta = linspace(min_a, max_a)

# Bot bottom and top of cone
ax.plot(min_r * cos(theta), min_r * sin(theta), 'k')
ax.plot(max_r * cos(theta), max_r * sin(theta), 'k')

# Plot sides of cone
ax.plot((min_r * cos(min_a), max_r * cos(min_a)), (min_r * sin(min_a), max_r * sin(min_a)), 'k')
ax.plot((min_r * cos(max_a), max_r * cos(max_a)), (min_r * sin(max_a), max_r * sin(max_a)), 'k')

# Plot connection to origin
ax.plot((0, min_r * cos(min_a)), (0, min_r * sin(min_a)), '--k')
ax.plot((0, min_r * cos(max_a)), (0, min_r * sin(max_a)), '--k')

# Constrain plot
ax.set_xlim((0, 3.25))
ax.set_ylim((0, 3.25))
ax.grid()

ax.tick_params(which='both', bottom=False, left=False, labelleft=False, labelbottom=False)

# Add axis labels
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')

# Save figure
dir = make_path('figures/thesis/')
make_dir(dir)
savefig(dir / 'refinement.png')

show()
