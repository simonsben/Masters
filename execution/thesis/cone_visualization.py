# from utilities.data_management import make_path
# from config import dataset
from utilities.plotting import generate_3d_figure
from matplotlib.pyplot import subplots, show
from numpy import linspace, pi, outer, sin, cos, ones_like, sqrt, meshgrid, arctan, tan, max, abs, deg2rad, zeros_like

fig, ax = generate_3d_figure()

# distance = 3
# base = (0, 0, 0)
# angle = 60
# d_min, d_max = 2, 5
#
# distances = linspace(0, distance, 250)
# polar_angles = linspace(0, 2 * pi, 250)
# radius_points, polar_points = meshgrid(distances, polar_angles)
#
# # Convert to cartesian
# x = radius_points * cos(polar_points) + base[0]
# y = radius_points * sin(polar_points) + base[1]
# z = radius_points / arctan(angle) + base[2]
#
#
# distances = sqrt(x ** 2 + y ** 2)
# too_small = distances < d_min
# too_big = distances > d_max
#
# z[too_small] = sqrt(d_min ** 2 + x[too_small] ** 2 + y[too_small] ** 2)
# # z[too_big] = d_max


def plot_limit(polar_angles, axis, slope, theta, limit):
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

d_min_limit = plot_limit(polar, ax, slope, theta, d_min)
d_max_limit = plot_limit(polar, ax, slope, theta, d_max)

plot_cone_walls(polar, ax, slope, theta, d_min, d_max)

# bot_r = d_min * cos(theta)
#
# bot_x = bot_r * cos(polar)
# bot_y = bot_r * sin(polar)
#
# bot_z = zeros_like(bot_x)
# bot_z[:] = d_min * sin(theta)

# ax.plot(bot_x, bot_y, bot_z)

# range = max((d_min_limit, d_max_limit))
# ax.set_xlim((-range, range))
# ax.set_ylim((-range, range))
# ax.set_zlim((0, d_max * 1.15))

show()
