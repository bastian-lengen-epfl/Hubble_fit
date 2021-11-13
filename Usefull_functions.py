### Useful functions
import numpy as np
from Values import *

# Wesenheit function
def wesenheit(data, R):
    return data['m_H'] - R * data['V-I']

# Distance modulus
def d(mu):
    return 10 ** (0.2 * mu - 5)

# Redshift magnitude relation
def redshift_magnitude_x(z):
    return np.log10(c * z * (1 + 1 / 2 * (1 - q_0) * z - 1 / 6 * (1 - q_0 - 3 * q_0 ** 2 + j_0) * z ** 2))
