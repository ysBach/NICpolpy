from photutils.aperture import CircularAnnulus, EllipticalAnnulus, RectangularAnnulus
import numpy as np

from . import STAR_1_2, STAR_1_2_err, RDN, GAIN
from ..background import annul2values, sky_fit

import pytest


@pytest.mark.parametrize(
    "positions, num1, num2, num500",
    [((24.5, 24.5), 38, 38, 0),
     ((10, 10), 0, 76, 0),
     ((10, 40), 76, 0, 0),
     ((35, 40), 76, 0, 0),
     ((35, 11), 0, 75, 1),
     ]
)
def test_annul2values_CircularAnnulus(positions, num1, num2, num500):
    an = CircularAnnulus(positions=positions, r_in=5, r_out=7)
    vals = annul2values(STAR_1_2, an, mask=None)
    assert len(vals[0]) == 76
    assert np.count_nonzero(vals[0] == 1) == num1
    assert np.count_nonzero(vals[0] == 2) == num2
    assert np.count_nonzero(vals[0] == 500) == num500
