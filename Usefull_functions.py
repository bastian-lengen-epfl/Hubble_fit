### Useful functions

# Wesenheit function
def wesenheit(data, R):
    return data['m_H'] - R * data['V-I']

# Distance modulus
def d(mu):
    return 10 ** (0.2 * mu - 5)
