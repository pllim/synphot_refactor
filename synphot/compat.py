# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Module to handle backward-compatibility."""

import astropy
import numpy
from astropy.utils.introspection import minversion

try:
    import specutils  # noqa
except ImportError:
    HAS_SPECUTILS = False
else:
    HAS_SPECUTILS = True

try:
    import dust_extinction  # noqa
except ImportError:
    HAS_DUST_EXTINCTION = False
else:
    HAS_DUST_EXTINCTION = True


__all__ = ['ASTROPY_LT_4_1', 'ASTROPY_LT_4_0', 'NUMPY_LT_1_17',
           'HAS_SPECUTILS', 'HAS_DUST_EXTINCTION']

ASTROPY_LT_4_1 = not minversion(astropy, '4.1')
ASTROPY_LT_4_0 = not minversion(astropy, '4.0')
NUMPY_LT_1_17 = not minversion(numpy, '1.17')
