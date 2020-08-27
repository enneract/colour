# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.tm_30_18` module.
"""

from __future__ import division, unicode_literals

import unittest
from matplotlib.pyplot import Figure, Axes

from colour.colorimetry import SDS_ILLUMINANTS
from colour.plotting.tm_30_18 import colour_rendition_report

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestColourRenditionReport']


class TestColourRenditionReport(unittest.TestCase):
    """
    Defines :func:`colour.plotting.tm_30_18.colour_rendition_report` definition
    unit tests methods.
    """

    def test_colour_rendition_report(self):
        """
        Tests :func:`colour.plotting.tm_30_18.colour_rendition_report`
        definition.
        """

        sd = SDS_ILLUMINANTS['FL8']

        source_information = {
            'source': 'CIE FL8',
            'date': 'Today',
            'manufacturer': 'CIE',
            'model': 'FL8',
            'notes': 'This is a standard CIE illuminant.'
        }

        for size in ['full', 'intermediate', 'simple']:
            figure, axes = colour_rendition_report(sd, size,
                                                   source_information)
            self.assertIsInstance(figure, Figure)
            self.assertIsInstance(axes, Axes)

        self.assertRaises(ValueError, colour_rendition_report, sd, 'bad size')
