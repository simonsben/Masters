from numpy import linspace, pi, outer, sin, cos, ones_like, meshgrid, arctan
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D


def plot_sphere(center, radius, axis=None, color='g', alpha=.6):
    if axis is None:
        fig = figure()
        axis = fig.add_subplot(111, projection='3d')

    # Define sphere
    u = linspace(0, pi)
    polar_angles = linspace(0, 2 * pi)

    # Convert to cartesian
    x = (radius * outer(sin(u), sin(polar_angles))) + center[0]
    y = (radius * outer(sin(u), cos(polar_angles))) + center[1]
    z = (radius * outer(cos(u), ones_like(polar_angles))) + center[2]

    axis.plot_surface(x, y, z, color=color, alpha=alpha)

    return axis


def plot_cone(base, angle, distance, axis=None, color='b', alpha=.6):
    if axis is None:
        fig = figure()
        axis = fig.add_subplot(111, projection='3d')

    distances = linspace(0, distance)
    polar_angles = linspace(0, 2 * pi)
    radius_points, polar_points = meshgrid(distances, polar_angles)

    x = radius_points * cos(polar_points) + base[0]
    y = radius_points * sin(polar_points) + base[1]
    z = radius_points / arctan(angle) + base[2]

    axis.plot_surface(x, y, z, color=color, alpha=alpha)

    return axis
