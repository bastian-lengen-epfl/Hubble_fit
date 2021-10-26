import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Values import *
from Usefull_functions import d

def Leavitt_correction(Cepheids: pd.DataFrame, SN: pd.DataFrame, galaxies: list):
    # Add z_obs to the Cepheids DataFrame
    z_obs = np.array([])
    for galaxy in galaxies:
        if galaxy == 'MW':
            z_obs = np.append(z_obs, np.zeros(len(Cepheids[Cepheids['Gal'] == galaxy])))
        elif galaxy == 'LMC':
            z_obs = np.append(z_obs, z_LMC * np.ones(len(Cepheids[Cepheids['Gal'] == galaxy])))
        elif galaxy == 'N4258':
            z_obs = np.append(z_obs, z_N4258 * np.ones(len(Cepheids[Cepheids['Gal'] == galaxy])))
        else:
            z_obs = np.append(z_obs, SN[SN['Gal'] == galaxy]['z_obs'].values[0] \
                              * np.ones(len(Cepheids[Cepheids['Gal'] == galaxy])))
    Cepheids['z_obs'] = z_obs

    # Correct the observed period P_obs to the absolute period P_0
    Cepheids['logP'] = Cepheids['logP'] - np.log10(1+Cepheids['z_obs'])

    return Cepheids

def RLB_galaxies_distance(pre_Leavitt_q: np.array, post_Leavitt_q: np.array, galaxies: list, name: str, fig_dir: str ='./Figure'):
    # Create the figure directory
    dist_dir = fig_dir + '/Distance'
    if not os.path.exists(dist_dir):
        print("I will create the %s directory for you." % dist_dir)
        os.mkdir(dist_dir)

    # Compute the difference in distance for each galaxy except MW
    d_obs = np.empty(len(galaxies) - 1)
    d_0 = np.empty(len(galaxies) - 1)
    for i in range(len(galaxies) - 1):
        d_obs[i] = d(pre_Leavitt_q[i])
        d_0[i] = d(post_Leavitt_q[i])

    ### plot it
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    ax.set_xlabel('d$_0$ [Mpc]', fontsize=18)
    ax.set_ylabel('d$_0$/d$_{obs}$ [Mpc]', fontsize=18)
    ax.invert_yaxis()
    colors = plt.cm.tab20(np.linspace(0, 1, len(d_0)))
    colors[-1] = [0, 0, 1, 1]
    for i in range(len(d_0)):
        if i in range(len(d_0) - 2, len(d_0)):  # Different sign for anchors
            ax.plot(d_0[i], d_0[i] / d_obs[i], marker='D', ms=15, ls='', label=galaxies[i], color=colors[i])
        else:
            ax.plot(d_0[i], d_0[i] / d_obs[i], marker='.', ms=30, ls='', label=galaxies[i], color=colors[i])
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    fig.savefig("%s/%s_distance.jpg"%(dist_dir,name), dpi=100)

    return

def Kcorr(Cepheids: pd.DataFrame):
    # First define the value of k according to the period and redshift
    def k(logP, z):
        # Reference points from Anderson 2021
        z_ref = np.array([0.0019, 0.0056, 0.0098, 0.0172, 0.0245])
        m = np.array([3.48, 2.68, 1.89, 1.07, 0.31]) * 1e-3
        c = np.array([0.51, 1.74, 3.25, 5.96, 8.05]) * 1e-3
        m_inter, c_inter, k = np.empty(len(z)), np.empty(len(z)), np.empty(len(z))

        # Linear interpolation
        for j in range(len(z)):
            for i in range(len(z_ref) - 1):
                if z_ref[i] <= z[j] and z[j] < z_ref[i + 1]:
                    m_inter[j] = m[i] + (z[j] - z_ref[i]) * (m[i + 1] - m[i]) / (z_ref[i + 1] - z_ref[i])
                    c_inter[j] = c[i] + (z[j] - z_ref[i]) * (c[i + 1] - c[i]) / (z_ref[i + 1] - z_ref[i])
                elif z[j] < z_ref[i]:
                    m_inter[j] = m[0] + (z[j] - z_ref[0]) * (m[1] - m[0]) / (z_ref[1] - z_ref[0])
                    c_inter[j] = c[0] + (z[j] - z_ref[0]) * (c[1] - c[0]) / (z_ref[1] - z_ref[0])
                else:
                    m_inter[j] = m[-2] + (z[j] - z_ref[-2]) * (m[-1] - m[-2]) / (z_ref[-1] - z_ref[-2])
                    c_inter[j] = c[-2] + (z[j] - z_ref[-2]) * (c[-1] - c[-2]) / (z_ref[-1] - z_ref[-2])
        return m_inter * logP + c_inter

    # Correct the magnitude of the Cepheids for the relativistics effect on k
    Cepheids['m_W'] = Cepheids['m_W'] \
                      + k(Cepheids['logP'], Cepheids['z_obs']) * Cepheids['z_obs'] * Cepheids['V-I'] \
                      - 0.105 * Cepheids['z_obs'] * Cepheids['V-I']  # for F99 redshift law

    return Cepheids

