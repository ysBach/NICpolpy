import numpy as np

STAR_1_2 = np.ones((50, 50))
STAR_1_2[:25, :] *= 2
STAR_1_2[23:27, 23:27] = np.array([
    [2, 1000, 1000, 2],
    [1000, 5000, 5000, 1000],
    [1000, 5000, 5000, 1000],
    [1, 1000, 1000, 1]
])
STAR_1_2[5::10, 40] = 500
STAR_1_2[15, 25] = 500
RDN = 2.5
GAIN = 2
STAR_1_2_err = np.sqrt(STAR_1_2/GAIN) + RDN/GAIN

