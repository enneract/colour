# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from .datasets import *  # noqa
from . import datasets
from .cfi2017 import CFI2017_Specification, colour_fidelity_index_CFI2017
from .cri import CRI_Specification, colour_rendering_index
from .cqs import (CQS_Specification, COLOUR_QUALITY_SCALE_METHODS,
                  colour_quality_scale)
from .ssi import spectral_similarity_index

__all__ = []
__all__ += datasets.__all__
__all__ += ['CFI2017_Specification', 'colour_fidelity_index_CFI2017']
__all__ += ['CRI_Specification', 'colour_rendering_index']
__all__ += [
    'CQS_Specification', 'COLOUR_QUALITY_SCALE_METHODS', 'colour_quality_scale'
]

__all__ += ['spectral_similarity_index']


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class quality(ModuleAPI):
    def __getattr__(self, attribute):
        return super(quality, self).__getattr__(attribute)


# v0.3.16
API_CHANGES = {
    'ObjectRenamed': [
        [
            'colour.quality.TCS_SDS',
            'colour.quality.SDS_TCS',
        ],
        [
            'colour.quality.VS_SDS',
            'colour.quality.SDS_VS',
        ],
    ]
}
"""
Defines *colour.quality* sub-package API changes.

API_CHANGES : dict
"""

if not is_documentation_building():
    sys.modules['colour.quality'] = quality(sys.modules['colour.quality'],
                                            build_API_changes(API_CHANGES))

    del ModuleAPI, is_documentation_building, build_API_changes, sys
