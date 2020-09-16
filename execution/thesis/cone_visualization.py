from utilities.plotting import generate_3d_figure
from matplotlib.pyplot import show
from numpy import linspace, pi, sin, cos, sqrt, meshgrid, tan, max, deg2rad

fig, ax = generate_3d_figure()


def plot_limit(polar_angles, axis, theta, limit):
    radius = linspace(0, limit * cos(theta))
    polar_angles, radius = meshgrid(polar_angles, radius)

    x = radius * cos(polar_angles)
    y = radius * sin(polar_angles)
    z = sqrt(limit ** 2 - (x ** 2 + y ** 2))

    axis.plot_surface(x, y, z, alpha=.8)

    return max(x)


def plot_cone_walls(polar, ax, slope, theta, d_min, d_max):
    cone_radius = linspace(d_min * cos(theta), d_max * cos(theta))
    cone_polar, cone_radius = meshgrid(polar, cone_radius)

    x = cone_radius * cos(cone_polar)
    y = cone_radius * sin(cone_polar)
    z = sqrt(x ** 2 + y ** 2) * slope

    ax.plot_surface(x, y, z, alpha=.4)


d_min, d_max = 2, 5

angle = 30
theta = deg2rad(90 - angle)
slope = tan(theta)

polar = linspace(0, 2 * pi)

d_min_limit = plot_limit(polar, ax, theta, d_min)
d_max_limit = plot_limit(polar, ax, theta, d_max)

plot_cone_walls(polar, ax, slope, theta, d_min, d_max)

show()
