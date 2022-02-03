### Plot functions :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Usefull_functions import redshift_magnitude_x


# Plot individual PL relation
def plot_PL(Cepheids: pd.DataFrame, Cepheids_Outliers: pd.DataFrame, galaxies: list, \
            q: list, name: str, fig_dir: str = "./Figure"):
    # Create the figure directory
    PL_dir = fig_dir + '/PL'
    if not os.path.exists(fig_dir):
        print("I will create the %s directory for you." % fig_dir)
        os.mkdir(fig_dir)
    if not os.path.exists(PL_dir):
        print("I will create the %s directory for you." % PL_dir)
        os.mkdir(PL_dir)

    # Function for individual PL plot
    def plot_ij(i, j, title):
        x_lim = [-0.5, 1.1]
        ax[i][j].errorbar(P, L, yerr=err, marker='.', ms=10, mfc="r", mec="k", ls="", c="k", lw=0.6)
        ax[i][j].errorbar(Pout, Lout, yerr=errout, marker='.', ms=10, mfc="lime", mec="k", ls="", c="k", lw=0.6)
        ax[i][j].invert_yaxis()
        ax[i][j].set_title(title, fontsize=12)
        ax[i][j].set_xlabel("log(P)-1", fontsize=8)
        ax[i][j].tick_params(axis='x', labelsize=8)
        if title == 'Milky Way':
            ax[i][j].set_ylabel("m_W - corrected (mag)", fontsize=8)
            ax[i][j].plot([x_lim[0], 0, x_lim[1]], [mW + bs * x_lim[0], mW, mW + bh * x_lim[1]])
        else:
            ax[i][j].set_ylabel("m_W - corrected (mag)", fontsize=8)
            ax[i][j].plot([x_lim[0], 0, x_lim[1]], [mW + bs * x_lim[0], mW, mW + bh * x_lim[1]] + mu)
        ax[i][j].set_xlim([-0.5, 1.1])
        y_lim = ax[i][j].get_ylim()
        ax[i][j].plot([0, 0], [y_lim[0], y_lim[1]], c='k', ls='--', lw=0.5)
        ax[i][j].set_ylim(y_lim)
        ax[i][j].tick_params(axis='y', labelsize=8)

    # Create the 5x5 grid
    fig, ax = plt.subplots(nrows=5, ncols=5)

    ax[3][4].remove()
    ax[4][3].remove()
    ax[4][4].remove()

    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.6)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    #  Load the parameters for the fit
    mW, bs, bh, ZW, zp = q[21], q[22], q[23], q[24], q[25]

    # Individual plot for each galaxy
    i, j = 0, 0
    for galaxy in galaxies:
        #  Prepare the x and y axis and plot it
        Filtered = Cepheids[Cepheids['Gal'] == galaxy]
        Filtered_out = Cepheids_Outliers[Cepheids_Outliers['Gal'] == galaxy]
        P = Filtered['logP'] - 1
        Pout = Filtered_out['logP'] - 1
        err = Filtered['sig_m_W']
        errout = Filtered_out['sig_m_W']
        if galaxy == 'MW':
            L = Filtered['m_W'] - 10 + 5 * np.log10(Filtered['pi']) \
                + 5 * zp / np.log(10) / Filtered['pi'] - ZW * Filtered['M/H']
            L_out = Filtered_out['m_W'] - 10 + 5 * np.log10(Filtered_out['pi']) \
                    + 5 * zp / np.log(10) / Filtered_out['pi'] - ZW * Filtered_out['M/H']
            mu = np.zeros(3)
            plot_ij(4, 0, galaxy)
        else:
            L = Filtered['m_W'] - ZW * Filtered['M/H']
            Lout = Filtered_out['m_W'] - ZW * Filtered_out['M/H']
            if galaxy == 'LMC':
                i, j = 4, 1
                mu = q[20]
            elif galaxy == 'N4258':
                i, j = 4, 2
                mu = q[19]
            else:
                mu = q[i * 5 + j]
            plot_ij(i, j, galaxy)

        #  Increments
        j = j + 1
        if j == 5:
            i = i + 1
            j = 0

    # Save the figure
    if len(Cepheids_Outliers) == 0:
        fig.savefig("%s/%s_individual.jpg" % (PL_dir, name), dpi=100)
    else:
        fig.savefig("%s/%s_individual.jpg" % (PL_dir, name), dpi=100)

    plt.close()
    return

# Global plot function
def plot_global_PL(Cepheids: pd.DataFrame, Cepheids_Outliers: pd.DataFrame, galaxies: list, \
                   q: list, name: str, fig_dir: str = "./Figure"):
    # Create the figure directory
    PL_dir = fig_dir + '/PL'
    if not os.path.exists(fig_dir):
        print("I will create the %s directory for you." % fig_dir)
        os.mkdir(fig_dir)
    if not os.path.exists(PL_dir):
        print("I will create the %s directory for you." % PL_dir)
        os.mkdir(PL_dir)

    #  Fit parameters
    mW, bs, bh, ZW, zp = q[21], q[22], q[23], q[24], q[25]

    # Scale all galaxies to theirs absolute magnitude
    logP1 = Cepheids['logP'] - 1
    logP1out = np.array([])  #  Cant load them all at once because we're iterating over the galaxies not the Cepheids
    L = np.array([])
    Lout = np.array([])
    i = 0
    for galaxy in galaxies:  # For hosts
        Filtered = Cepheids[Cepheids['Gal'] == galaxy]
        Filtered_out = Cepheids_Outliers[Cepheids_Outliers['Gal'] == galaxy]
        if galaxy == 'MW':
            logP1out = np.append(logP1out, Filtered_out['logP'] - 1)
            L = np.append(L, Filtered['m_W'] - 10 + 5 * np.log10(Filtered['pi']) \
                          + 5 * zp / np.log(10) / Filtered['pi'] - ZW * Filtered['M/H'])
            Lout = np.append(Lout, Filtered_out['m_W'] - 10 + 5 * np.log10(Filtered_out['pi']) \
                             + 5 * zp / np.log(10) / Filtered_out['pi'] - ZW * Filtered_out['M/H'])
        else:
            mu = q[i]
            logP1out = np.append(logP1out, Filtered_out['logP'] - 1)
            L = np.append(L, Filtered['m_W'] - mu - ZW * Filtered['M/H'])
            Lout = np.append(Lout, Filtered_out['m_W'] - mu - ZW * Filtered_out['M/H'])
            i = i + 1

    #  Create the figure
    fig, ax = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [7, 3]})
    fig.set_figheight(10)
    fig.set_figwidth(15)

    # Top panel
    ax[0].set_title('Global PL relation for the absolute magnitude', fontsize=16)
    ax[0].plot(logP1, L, marker='.', ms=12, mfc="r", mec="k", ls="", c="k", lw=3)
    ax[0].plot(logP1out, Lout, marker='.', ms=12, mfc="lime", mec="k", ls="", c="k", lw=3)
    xmin, xmax = ax[0].get_xlim()
    ymin, ymax = ax[0].get_ylim()
    ax[0].plot([xmin, 0, xmax], [mW + bs * xmin, mW, mW + bh * xmax], c='tab:blue', ls='-', lw=3)
    ax[0].plot([0, 0], [ymin, ymax], c='k', ls='--', lw=1.5)
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    ax[0].invert_yaxis()
    ax[0].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    ax[0].set_ylabel('M$_W$ corrected [mag]', fontsize=14)

    # Bottom panel
    error = np.zeros(len(logP1))
    for i in range(0, np.size(logP1)):
        if logP1[i] < 0:
            error[i] = L[i] - mW - bs * logP1[i]
        else:
            error[i] = L[i] - mW - bh * logP1[i]
    errorout = np.zeros(len(logP1out))
    for i in range(0, len(logP1out)):
        if logP1out[i] < 0:
            errorout[i] = Lout[i] - mW - bs * logP1out[i]
        else:
            errorout[i] = Lout[i] - mW - bh * logP1out[i]

    ax[1].plot(logP1, error, marker='D', ms=5, mfc="none", mec="firebrick", ls="", c="k", lw=0)
    ax[1].plot(logP1out, errorout, marker='D', ms=5, mfc="none", mec="lime", ls="", c="k", lw=0)
    ymin, ymax = ax[1].get_ylim()
    ax[1].plot([xmin, xmax], [0, 0], c='k', ls='--', lw=1.5)
    ax[1].plot([0, 0], [ymin, ymax], c='k', ls='--', lw=1.5)
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    ax[1].invert_yaxis()
    ax2 = ax[1].twiny()  # ax1 and ax2 share y-axis
    ax2.plot(logP1, error, '.', markersize=0)
    ax[1].set_xlabel('logP-1', fontsize=14)
    ax[1].set_ylabel('$\Delta$M$_W$ [mag]', fontsize=14)

    # Save figures
    fig.subplots_adjust(wspace=0, hspace=0)
    if len(Cepheids_Outliers) == 0:
        fig.savefig("%s/%s_global.jpg" % (PL_dir, name), dpi=100)
    else:
        fig.savefig("%s/%s_global.jpg" % (PL_dir, name), dpi=100)

    plt.close()
    return

# Hubble plot
def Hubble_plot(Hubble: pd.DataFrame, fig_dir: str = "./Figure"):
    # Get the color and marker depending on the category Cepheids/TRGB/Avg
    category = Hubble['C/T/Avg']
    color = [None]*len(category)
    marker = [None]*len(category)
    for i in range(len(category)):
        if category[i] == 'C':
            color[i] = 'tab:blue'
            marker[i] = 'o'
        elif category[i] == 'T':
            color[i] = 'tab:orange'
            marker[i] = '^'
        else:
            color[i] = 'tab:green'
            marker[i] = 's'

    # Get the name
    names = list(Hubble['Sim'])+ ['Riess 2021', 'Anand 2021']
    ### Plot Hubble's constant
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    for i in range(len(Hubble)):
        ax.errorbar(i, Hubble.loc[i,'H_0'], Hubble.loc[i,'sig_H_0'], marker=marker[i], color=color[i], ms=18, ls='')
    ax.errorbar([i+1, i+2], [73.2, 71.5], [1.1,1.8], marker='X', ms=18, c='r', ls='')
    ax.set_ylabel('H$_0$ [Mpc]', fontsize=18)
    ax.set_xticklabels(names, rotation=90, minor=True)
    ax.tick_params(labelsize=10)
    fig.savefig('%s/Hubble.jpg'%fig_dir, dpi=100)

# SN plot
def plot_SN(a_b: float, fit_values: list, z_min: float = 0.023, z_max: float = 0.15, fig_dir: str = "./Figure"):
    [z, dz, m, dm, z_fit, dz_fit, m_fit, dm_fit] = fit_values

    ### Do the plot
    fig, ax = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [3, 1]})

    ax[0].plot(redshift_magnitude_x(z), 0.2 * m, marker='.', ms=12, mfc="lime", mec="k", ls="", c="k", lw=3)
    ax[0].plot(redshift_magnitude_x(z_fit), 0.2 * m_fit, marker='.', ms=12, mfc="r", mec="k", ls="", c="k", lw=3)
    tmp = ax[0].get_xlim()
    ax[0].plot(tmp, np.array(tmp) - a_b, 'k', lw=2)  # slope 1 by mean
    ax[0].set_xlim(tmp)
    tmp = ax[0].get_ylim()
    ax[0].plot(redshift_magnitude_x(np.array([z_min, z_min])), tmp, c='k', ls='--', lw=1)
    ax[0].text(redshift_magnitude_x(z_min) + 0.05, tmp[1] - 0.15, 'z=%f' % z_min,  size = 16)
    ax[0].plot(redshift_magnitude_x(np.array([z_max, z_max])), tmp, c='k', ls='--', lw=1)
    ax[0].text(redshift_magnitude_x(z_max) + 0.05, tmp[1] - 0.15, 'z=%f' % z_max, size = 16)
    ax[0].set_ylim(tmp)
    ax[0].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    ax[0].set_ylabel('0.2m$_b$ [mag]', size = 16)
    ax[0].tick_params(axis='both', which='major', labelsize=12)

    ax[1].plot(redshift_magnitude_x(z), 0.2 * m - (redshift_magnitude_x(z) - a_b), \
               marker='D', ms=5, mfc="none", mec="limegreen", ls="", c="k", lw=0)
    ax[1].plot(redshift_magnitude_x(z_fit), 0.2 * m_fit - (redshift_magnitude_x(z_fit) - a_b), \
               marker='D', ms=5, mfc="none", mec="r", ls="", c="k", lw=0)
    tmp = [ax[0].get_xlim()[0], ax[0].get_xlim()[1]]
    ax[1].plot(tmp, [0, 0], c='k', ls='--', lw=3)
    ax[1].set_xlim(tmp)
    tmp = ax[1].get_ylim()
    ax[1].plot(redshift_magnitude_x(np.array([z_min, z_min])), tmp, c='k', ls='--', lw=1)
    ax[1].plot(redshift_magnitude_x(np.array([z_max, z_max])), tmp, c='k', ls='--', lw=1)
    ax[1].set_ylim(tmp)
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    ax2 = ax[1].twiny()  # ax1 and ax2 share y-axis
    ax2.plot(redshift_magnitude_x(z), 0.2 * m - (redshift_magnitude_x(z) - a_b), '.', markersize=0)
    ax[1].set_xlabel('log{cz[1+0.5(1-q$_0$)z-(1/6)(1-q$_0$-3q$_0^2$+1)z$^2$]}', size=16)
    ax[1].set_ylabel('$\Delta$0.2m$_b$ [mag]', size=16)

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    fig.savefig('%s/SN_Pantheon.jpg'%fig_dir, dpi=100)

    return