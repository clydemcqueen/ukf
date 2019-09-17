#!/usr/bin/env python3

"""
Plot the output from test_1d_drag_filter()
"""

import sys

import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        print('Usage: plot_ukf_1d_filter.py filename')
        exit(1)

    f = open(sys.argv[1], 'r')
    print('Expecting t, actual_x, actual_vx, actual_ax, z, x.x, x.vx, x.ax, K.x, K.vx, K.ax')
    print(f.readline())

    actual_xs = []
    actual_vxs = []
    actual_axs = []
    zs = []
    xs = []
    vxs = []
    axs = []
    kxs = []
    kvxs = []
    kaxs = []

    for line in f:
        fields = line.split(',')
        actual_xs.append(float(fields[1]))
        actual_vxs.append(float(fields[2]))
        actual_axs.append(float(fields[3]))
        zs.append(float(fields[4]))
        xs.append(float(fields[5]))
        vxs.append(float(fields[6]))
        axs.append(float(fields[7]))
        kxs.append(float(fields[8]))
        kvxs.append(float(fields[9]))
        kaxs.append(float(fields[10]))

    fig, (plot_x, plot_vx, plot_ax, plot_k) = plt.subplots(4, 1)

    plot_x.plot(actual_xs, label='actual_x')
    plot_x.plot(zs, marker='x', ls='', label='z')
    plot_x.plot(xs, label='x')

    plot_vx.plot(actual_vxs, label='actual_vx')
    plot_vx.plot(vxs, label='vx')

    plot_ax.plot(actual_axs, label='actual_ax')
    plot_ax.plot(axs, label='ax')

    plot_k.plot(kxs, label='kalman x')
    plot_k.plot(kvxs, label='kalman vx')
    plot_k.plot(kaxs, label='kalman ax')

    plot_x.legend()
    plot_vx.legend()
    plot_ax.legend()
    plot_k.legend()

    print('Click plot to exit')
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
