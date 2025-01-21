import warnings

import numpy as np
from astropy.wcs import FITSFixedWarning

from .NICPolReduc import *
from .phot import *
from .prepare import *
from .preproc import *
from .util import *

np.set_printoptions(legacy="1.25")

warnings.filterwarnings('ignore', append=True, category=FITSFixedWarning)
