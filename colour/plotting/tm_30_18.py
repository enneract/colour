# -*- coding: utf-8 -*-
"""
ANSI/IES TM-30-18 Colour Rendition Report
=========================================

Defines the colour quality plotting objects:

...
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import colour
from colour.colorimetry import sd_to_XYZ
from colour.plotting import override_style, render
from colour.models import XYZ_to_xy, XYZ_to_Luv, Luv_to_uv
from colour.quality import (colour_rendering_index,
                            colour_fidelity_index_TM_30_18)
from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['colour_rendition_report']

CVG_BACKGROUND_PATH = os.path.join(
    os.path.dirname(__file__),
    'resources/Colour vector graphic background.jpg')
"""
Path to the background image used in a *Colour Vector Graphic*.

CVG_BACKGROUND_PATH : unicode
"""

BIN_BAR_COLOURS = [
    '#a35c60', '#cc765e', '#cc8145', '#d8ac62', '#ac9959', '#919e5d',
    '#668b5e', '#61b290', '#7bbaa6', '#297a7e', '#55788d', '#708ab2',
    '#988caa', '#735877', '#8f6682', '#ba7a8e'
]
"""
*RGB* values for representing hue-angle bins in bar charts, specified in Annex
B of the standard.

BIN_BAR_COLORS : list of str
"""

BIN_ARROW_COLOURS = [
    '#e62828', '#e74b4b', '#fb812e', '#ffb529', '#cbca46', '#7eb94c',
    '#41c06d', '#009c7c', '#16bcb0', '#00a4bf', '#0085c3', '#3b62aa',
    '#4568ae', '#6a4e85', '#9d69a1', '#a74f81'
]
"""
*RGB* values for representing hue-angle bin vectors in the *Colour Vector
Graphic*, specified in Annex B of the standard.

BIN_ARROW_COLOURS : list of str
"""

TCS_BAR_COLOURS = [
    '#f1bdcd', '#ca6183', '#573a40', '#cd8791', '#ad3f55', '#925f62',
    '#933440', '#8c3942', '#413d3e', '#fa8070', '#c35644', '#da604a',
    '#824e39', '#bca89f', '#c29a89', '#8d593c', '#915e3f', '#99745b',
    '#d39257', '#d07f2c', '#feb45f', '#efa248', '#f0dfbd', '#fed586',
    '#d0981e', '#fed06a', '#b5ac81', '#645d37', '#ead163', '#9e9464',
    '#ebd969', '#c4b135', '#e6de9c', '#99912c', '#61603a', '#c2c2af',
    '#6d703b', '#d2d7a1', '#4b5040', '#6b7751', '#d3dcc3', '#88b33a',
    '#8ebf3e', '#3e3f3d', '#65984a', '#83a96e', '#92ae86', '#91cd8e',
    '#477746', '#568c6a', '#659477', '#276e49', '#008d62', '#b6e2d4',
    '#a5d9cd', '#39c4ad', '#00a18a', '#009786', '#b4e1d9', '#cddddc',
    '#99c1c0', '#909fa1', '#494d4e', '#009fa8', '#32636a', '#007788',
    '#007f95', '#66a0b2', '#687d88', '#75b6db', '#1e5574', '#aab9c3',
    '#3091c4', '#3b3e41', '#274d72', '#376fb8', '#496692', '#3b63ac',
    '#a0aed5', '#9293c8', '#61589d', '#d4d3e5', '#aca6ca', '#3e3b45',
    '#5f5770', '#a08cc7', '#664782', '#a77ab5', '#6a4172', '#7d4983',
    '#c4bfc4', '#937391', '#ae91aa', '#764068', '#bf93b1', '#d7a9c5',
    '#9d587f', '#ce6997', '#ae4a79'
]
"""
*RGB* values of the 99 *test colour samples*, illuminated by *CIE D65*, in the
*sRGB* colourspace.

TCS_BAR_COLOURS: list of str
"""


def plot_sd_comparison(ax, spec):
    """
    Plots a comparison of spectral distributions of a test and a reference
    illuminant, for use in *TM-30-18* colour rendition reports.

    Parameters
    ----------
    ax : Axes
        Axes to plot on.
    spec : TM_30_18_Specification
        *TM-30-18* colour fidelity specification.
    """

    Y_reference = sd_to_XYZ(spec.sd_reference)[1]
    Y_test = sd_to_XYZ(spec.sd_test)[1]

    ax.plot(
        spec.sd_reference.wavelengths,
        spec.sd_reference.values / Y_reference,
        'k',
        label='Reference')
    ax.plot(
        spec.sd_test.wavelengths,
        spec.sd_test.values / Y_test,
        'r',
        label='Test')

    ax.set_yticks([])
    ax.grid()

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Radiant power')
    ax.legend()


def plot_colour_vector_graphic(ax, spec):
    """
    Plots a *Colour Vector Graphic* according to *TM-30-18* recommendations.

    Parameters
    ----------
    ax : Axes
        Axes to plot on.
    spec : TM_30_18_Specification
        *TM-30-18* colour fidelity specification.
    """

    # Background
    backdrop = mpimg.imread(CVG_BACKGROUND_PATH)
    ax.imshow(backdrop, extent=(-1.5, 1.5, -1.5, 1.5))
    ax.axis('off')

    # Lines dividing the hues in 16 equal parts along with bin numbers
    ax.plot(0, 0, '+', color='#a6a6a6')
    for i in range(16):
        angle = 2 * np.pi * i / 16
        dx = np.cos(angle)
        dy = np.sin(angle)
        ax.plot(
            (0.15 * dx, 1.5 * dx), (0.15 * dy, 1.5 * dy),
            '--',
            color='#a6a6a6',
            lw=0.75)

        angle = 2 * np.pi * (i + 0.5) / 16
        ax.annotate(
            str(i + 1),
            color='#a6a6a6',
            ha='center',
            va='center',
            xy=(1.41 * np.cos(angle), 1.41 * np.sin(angle)),
            weight='bold',
            size=9)

    # Circles
    circle = plt.Circle((0, 0), 1, color='black', lw=1.25, fill=False)
    ax.add_artist(circle)
    for radius in [0.8, 0.9, 1.1, 1.2]:
        circle = plt.Circle((0, 0), radius, color='white', lw=0.75, fill=False)
        ax.add_artist(circle)

    # -/+20% marks near the white circles
    props = dict(ha='right', color='white', size=7)
    ax.annotate('-20%', xy=(0, -0.8), va='bottom', **props)
    ax.annotate('+20%', xy=(0, -1.2), va='top', **props)

    # Average CAM02 h correlate for each bin, in radians
    average_hues = as_float_array([
        np.mean([spec.colorimetry_data[1][i].CAM.h for i in spec.bins[j]])
        for j in range(16)
    ]) / 180 * np.pi
    xy_reference = np.vstack([np.cos(average_hues), np.sin(average_hues)]).T

    # Arrow offsets as defined by the standard
    offsets = ((spec.averages_test - spec.averages_reference) /
               spec.average_norms[:, np.newaxis])
    xy_test = xy_reference + offsets

    # Arrows
    for i in range(16):
        ax.arrow(
            xy_reference[i, 0],
            xy_reference[i, 1],
            offsets[i, 0],
            offsets[i, 1],
            length_includes_head=True,
            width=0.005,
            head_width=0.04,
            linewidth=None,
            color=BIN_ARROW_COLOURS[i])

    # Red (test) gamut shape
    loop = np.append(xy_test, xy_test[0, np.newaxis], axis=0)
    ax.plot(loop[:, 0], loop[:, 1], '-', color='#f05046', lw=2)

    def corner_text(label, text, ha, va):
        x = -1.44 if ha == 'left' else 1.44
        y = 1.44 if va == 'top' else -1.44
        y_text = -14 if va == 'top' else 14

        ax.annotate(
            text,
            xy=(x, y),
            color='black',
            ha=ha,
            va=va,
            weight='bold',
            size=14)
        ax.annotate(
            label,
            xy=(x, y),
            color='black',
            xytext=(0, y_text),
            textcoords='offset points',
            ha=ha,
            va=va,
            size=11)

    corner_text('$R_f$', '{:.0f}'.format(spec.R_f), 'left', 'top')
    corner_text('$R_g$', '{:.0f}'.format(spec.R_g), 'right', 'top')
    corner_text('CCT', '{:.0f} K'.format(spec.CCT), 'left', 'bottom')
    corner_text('$D_{uv}$', '{:.4f}'.format(spec.D_uv), 'right', 'bottom')


def _plot_bin_bars(ax, values, ticks, labels_format, labels='vertical'):
    """
    A convenience function for plotting coloured bar graphs with labels at each
    bar.
    """

    ax.set_axisbelow(True)  # Draw the grid behind the bars
    ax.grid(axis='y')

    ax.bar(np.arange(16) + 1, values, color=BIN_BAR_COLOURS)
    ax.set_xlim(0.5, 16.5)
    if ticks:
        ax.set_xticks(np.arange(1, 17))
    else:
        ax.set_xticks([])

    for i, value in enumerate(values):
        if labels == 'vertical':
            va = 'bottom' if value > 0 else 'top'
            ax.annotate(
                labels_format.format(value),
                xy=(i + 1, value),
                rotation=90,
                ha='center',
                va=va)

        elif labels == 'horizontal':
            va = 'bottom' if value < 90 else 'top'
            ax.annotate(
                labels_format.format(value),
                xy=(i + 1, value),
                ha='center',
                va=va)


def plot_local_chroma_shifts(ax, spec, ticks=True):
    """
    Creates a bar graph of local chroma shifts.

    Parameters
    ----------
    ax : Axes
        Axes to plot on.
    spec : TM_30_18_Specification
        *TM-30-18* colour fidelity specification.
    ticks : bool, optional
        Whether to include ticks and labels on the X axis.
    """

    _plot_bin_bars(ax, spec.R_cs, ticks, '{:.0f}%')
    ax.set_ylim(-40, 40)
    ax.set_ylabel('Local Chroma Shift ($R_{cs,hj}$)')

    ticks = np.arange(-40, 41, 10)
    ax.set_yticks(ticks)
    ax.set_yticklabels(['{}%'.format(value) for value in ticks])


def plot_local_hue_shifts(ax, spec, ticks=True):
    """
    Creates a bar graph of local hue hifts.

    Parameters
    ----------
    ax : Axes
        Axes to plot on.
    spec : TM_30_18_Specification
        *TM-30-18* colour fidelity specification.
    ticks : bool, optional
        Whether to include ticks and labels on the X axis.
    """

    _plot_bin_bars(ax, spec.R_hs, ticks, '{:.2f}')
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks(np.arange(-0.5, 0.51, 0.1))
    ax.set_ylabel('Local Hue Shift ($R_{hs,hj}$)')


def plot_local_colour_fidelities(ax, spec, ticks=True):
    """
    Creates a bar graph of local colour fidelities.

    Parameters
    ----------
    ax : Axes
        Axes to plot on.
    spec : TM_30_18_Specification
        *TM-30-18* colour fidelity specification.
    ticks : bool, optional
        Whether to include ticks and labels on the X axis.
    """

    _plot_bin_bars(ax, spec.R_fs, ticks, '{:.0f}', 'horizontal')
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xlabel('Hue-angle bin')
    ax.set_ylabel('Local Colour Fidelity ($R_{f,hj}$)')


def plot_colour_sample_fidelities(ax, spec):
    """
    Creates a dense bar graph of colour sample fidielities.

    Parameters
    ----------
    ax : Axes
        Axes to plot on.
    spec : TM_30_18_Specification
        *TM-30-18* colour fidelity specification.
    """
    ax.set_axisbelow(True)  # Draw the grid behind the bars
    ax.grid(axis='y')

    ax.bar(np.arange(99) + 1, spec.Rs, color=TCS_BAR_COLOURS)
    ax.set_xlim(0.5, 99.5)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel('Colour Sample Fidelity ($R_{f,ces}$)')

    ticks = list(range(1, 100, 1))
    ax.set_xticks(ticks)

    labels = [
        'CES{:02d}'.format(i) if i % 3 == 1 else '' for i in range(1, 100)
    ]
    ax.set_xticklabels(labels, rotation=90)


def full_report(spec, source_information):
    """
    Creates a figure with a full *TM-30-18* colour rendition report.

    Parameters
    ----------
    spec : TM_30_18_Specification
        *TM-30-18* colour fidelity specification.
    source_information : dict
        Dictionary containing information about the tested light source.
    Returns
    -------
    Figure
        Figure object.
    """

    figure = plt.figure(figsize=(8.27, 11.69))

    figure.text(
        0.5,
        0.97,
        'TM-30-18 Color Rendition Report',
        ha='center',
        size='x-large')

    figure.text(
        0.250, 0.935, 'Source: ', ha='right', size='large', weight='bold')
    figure.text(0.250, 0.935, source_information['source'], size='large')
    figure.text(
        0.250, 0.907, 'Date: ', ha='right', size='large', weight='bold')
    figure.text(0.250, 0.907, source_information['date'], size='large')

    figure.text(
        0.700,
        0.935,
        'Manufacturer: ',
        ha='right',
        size='large',
        weight='bold')
    figure.text(
        0.700,
        0.935,
        source_information['manufacturer'],
        ha='left',
        size='large')
    figure.text(
        0.700, 0.907, 'Model: ', ha='right', size='large', weight='bold')
    figure.text(0.700, 0.907, source_information['model'], size='large')

    ax = figure.add_axes((0.057, 0.767, 0.385, 0.112))
    plot_sd_comparison(ax, spec)

    ax = figure.add_axes((0.036, 0.385, 0.428, 0.333))
    plot_colour_vector_graphic(ax, spec)

    ax = figure.add_axes((0.554, 0.736, 0.409, 0.141))
    plot_local_chroma_shifts(ax, spec)

    ax = figure.add_axes((0.554, 0.576, 0.409, 0.141))
    plot_local_hue_shifts(ax, spec)

    ax = figure.add_axes((0.554, 0.401, 0.409, 0.141))
    plot_local_colour_fidelities(ax, spec)

    ax = figure.add_axes((0.094, 0.195, 0.870, 0.161))
    plot_colour_sample_fidelities(ax, spec)

    figure.text(
        0.139, 0.114, 'Notes: ', ha='right', size='large', weight='bold')
    figure.text(0.139, 0.114, source_information['notes'], size='large')

    XYZ = sd_to_XYZ(spec.sd_test)
    xy = XYZ_to_xy(XYZ)
    Luv = XYZ_to_Luv(XYZ, xy)
    up, vp = Luv_to_uv(Luv, xy)

    figure.text(0.712, 0.111, '$x$  {:.4f}'.format(xy[0]), ha='center')
    figure.text(0.712, 0.091, '$y$  {:.4f}'.format(xy[1]), ha='center')
    figure.text(0.712, 0.071, '$u\'$  {:.4f}'.format(up), ha='center')
    figure.text(0.712, 0.051, '$v\'$  {:.4f}'.format(vp), ha='center')

    rect = plt.Rectangle(
        (0.814, 0.035), 0.144, 0.096, color='black', fill=False)
    figure.add_artist(rect)

    CRI_spec = colour_rendering_index(spec.sd_test, additional_data=True)

    figure.text(0.886, 0.111, 'CIE 13.31-1995', ha='center')
    figure.text(0.886, 0.091, '(CRI)', ha='center')
    figure.text(
        0.886,
        0.071,
        '$R_a$  {:.0f}'.format(CRI_spec.Q_a),
        ha='center',
        weight='bold')
    figure.text(
        0.886,
        0.051,
        '$R_9$  {:.0f}'.format(CRI_spec.Q_as[8].Q_a),
        ha='center',
        weight='bold')

    figure.text(
        0.500, 0.010, 'Created with Colour ' + colour.__version__, ha='center')

    return figure


def intermediate_report(spec):
    """
    Creates a figure with a full *TM-30-18* colour rendition report.

    Parameters
    ----------
    spec : TM_30_18_Specification
        *TM-30-18* colour fidelity specification.

    Returns
    -------
    Figure
        Figure object.
    """

    figure = plt.figure(figsize=(8.27, 4.44))

    figure.text(
        0.500,
        0.945,
        'TM-30-18 Color Rendition Report',
        ha='center',
        size='x-large')

    ax = figure.add_axes((0.024, 0.077, 0.443, 0.833))
    plot_colour_vector_graphic(ax, spec)

    ax = figure.add_axes((0.560, 0.550, 0.409, 0.342))
    plot_local_chroma_shifts(ax, spec)

    ax = figure.add_axes((0.560, 0.150, 0.409, 0.342))
    plot_local_hue_shifts(ax, spec)

    figure.text(
        0.500, 0.020, 'Created with Colour ' + colour.__version__, ha='center')

    return figure


def simple_report(spec):
    """
    Creates a figure with a simple *TM-30-18* colour rendition report.

    Parameters
    ----------
    spec : TM_30_18_Specification
        *TM-30-18* colour fidelity specification.

    Returns
    -------
    Figure
        Figure object.
    """

    figure = plt.figure(figsize=(4.22, 4.44))

    figure.text(
        0.500,
        0.945,
        'TM-30-18 Color Rendition Report',
        ha='center',
        size='x-large')

    ax = figure.add_axes((0.05, 0.05, 0.90, 0.90))
    plot_colour_vector_graphic(ax, spec)

    figure.text(
        0.500, 0.022, 'Created with Colour ' + colour.__version__, ha='center')

    return figure


@override_style()
def colour_rendition_report(sd, size='full', source_information=None,
                            **kwargs):
    """
    Create a *TM-30-18* color rendition report, according to rules and
    recommendations specified in the standard.

    Parameters
    ----------
    sd : SpectralDistribution
        Spectral power distribution of the light source.
    size : unicode, optional
        Size of the report. The three sizes recommended by the standard are
        supported:

        - ``'full'`` (the default), a comprehensive report, contaning:
            - test
            - 123
            -test
        - aaa

    source_information : dict, optional
        Dictionary containing information about the tested light source.
        It is ignored unless

        source : unicode
            Light source name.
        date : unicode
            Report creation date.
        manufacturer : unicode
            Light source manufacturer.
        model : unicode
            Light source model name.
        notes : unicode
            Notes that will appear at the bottom of the report.
    """

    spec = colour_fidelity_index_TM_30_18(sd, additional_data=True)

    if size == 'full':
        figure = full_report(spec, source_information)
    elif size == 'intermediate':
        figure = intermediate_report(spec)
    elif size == 'simple':
        figure = simple_report(spec)
    else:
        raise ValueError('size must be one of \'simple\', \'intermediate\' or '
                         '\'full\'')

    settings = {'figure': figure}
    settings.update(kwargs)

    return render(**settings)
