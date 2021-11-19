### Create and load the differents dataframe used
import numpy as np
from astropy.io import fits
from astropy.table import Table
import pandas as pd

from Usefull_functions import wesenheit
from Values import *


def load_Cepheids(data_dir: str = './Data'):
    # Create empty Cepheids DataFrame
    Cepheids = pd.DataFrame({'Gal': [], 'logP': [], 'm_W': [], 'sig_m_W': [],\
                             'M/H': [], 'pi': [], 'sig_pi': [], 'V-I': []})

    # Load Hosts + M31 + N4258 Cepheids from Riess 2016
    tmp = pd.read_csv('%s/Data_Riess2016/Cepheids.csv'%data_dir).sort_values(by="recno")\
            .set_index("recno").rename(columns={'F160W':'m_H'})
    tmp['m_W'] = wesenheit(tmp,R) # Add wesenheit magnitude
    tmp = tmp.drop(tmp[tmp['Gal']=='M31'].index) # Drop M31
    Cepheids = Cepheids.append(pd.DataFrame({'Gal':tmp['Gal'], 'logP':np.log10(tmp['Per']), 'm_W':tmp['m_W'],\
                                             'sig_m_W':tmp['sigTot'], 'M/H':tmp['[O/H]']-Z_sun, 'pi':np.NaN,\
                                             'sig_pi':np.NaN, 'V-I':tmp['V-I']}), ignore_index = True)

    # Load the LMC Cepheids (Riess 2019)
    with fits.open('%s/Data_Riess2019/Cepheids_LMC.fit'%data_dir) as hdul:
        tmp = Table(hdul[1].data)
    Cepheids = Cepheids.append(pd.DataFrame({'Gal':'LMC', 'logP':tmp['logP'], 'm_W':tmp['mWH'],\
                                'sig_m_W':tmp['e_mWH'], 'M/H': -0.30, 'pi':np.NaN,\
                                'sig_pi':np.NaN,'V-I':tmp['F555Wmag']-tmp['F814Wmag']}), ignore_index = True)

    # Load MW Cepheids(Riess 2021)
    tmp = pd.read_csv('%s/Data_Riess2021/Cepheids_MW.csv'%data_dir, sep = ',')
    tmp = tmp.drop(tmp[tmp['pi_EDR3']=='⋯'].index) # Drop cepheids with no pi
    tmp['pi_EDR3'] = np.array(tmp['pi_EDR3'], dtype=float) # str -> float
    tmp['sig_pi_EDR3'] = np.array(tmp['sig_pi_EDR3'], dtype=float) # str -> float
    Cepheids = Cepheids.append(pd.DataFrame({'Gal':'MW', 'logP':tmp['logP'], 'm_W':tmp['m_W'],\
                                'sig_m_W':tmp['sig_m_W'], 'M/H':tmp['[Fe/H]'], 'pi':tmp['pi_EDR3'],\
                                'sig_pi':tmp['sig_pi_EDR3'], 'V-I':tmp['V']-tmp['I']}), ignore_index = True)

    # Load the SN data (Riess 2016)
    SN = pd.read_csv('%s/Data_Riess2016/Sn.csv'%data_dir)
    SN["recno"]=np.array([1,2,3,4,5,6,7,8,9,10,11,12,20,13,14,15,16,17,18,19])-1 #re-order
    SN = SN.sort_values(by="recno").set_index("recno") # re-index
    SN.index.name = None
    SN = SN[0:19] ### Drop m31 gal with no supernova
    SN['m_b'] = np.array([13.310, 17.015, 16.756, 15.482, 15.765, 15.840, 16.527, 16.476, 16.265, 16.048,\
                          15.795, 15.797, 15.110, 15.177, 15.983, 16.265, 16.572, 15.867, 17.034])-5*a_b
    SN['sig'] = np.array([0.117, 0.123, 0.116, 0.125, 0.116, 0.142, 0.117, 0.115, 0.124, 0.116, 0.115, 0.114,\
                          0.109, 0.124, 0.115, 0.115, 0.115, 0.115, 0.114]) ### Table 5 Riess 2016
    SN['sig'] = np.sqrt(SN['sig']**2-(5*sigma_a_b)**2) ### to have sigma from magnitude and not total sigma
    SN = SN.drop(['f_Gal', 'IRexp','Oexp','PID', 'Date', '_RA', '_DE'], axis=1)
    SN['z_obs'] = 1e-3*np.array([0.80, 8.77, 7.12, 5.45,  3.90, 4.89, 5.14, 4.27, 3.56, 2.84, 3.70, 5.48, 1.46,\
                            6.03, 3.40, 5.46, 6.35, 3.89, 6.38]) # p.5 Anderson 2019


    # list of galaxies
    galaxies = Cepheids.Gal.drop_duplicates().reset_index()['Gal']

    return Cepheids, SN, galaxies

def load_SN_pantheon(data_dir: str = './Data'):
    #  Load the data for supernovae m_b - redshift plot (Pantheon)
    SN_Pantheon = pd.read_csv('%s/Data_Pantheon/SN.txt' % data_dir, sep=' ')
    SN_Pantheon = SN_Pantheon.drop(columns=['x1', 'dx1', 'color', 'dcolor', '3rdvar', 'd3rdvar', 'cov_m_s', \
                                            'cov_m_c', 'cov_s_c', 'set', 'ra', 'dec', 'biascor'])
    return SN_Pantheon

def load_TRGB(data_dir: str = './Data'):
    # Import data and split them in two dataframe
    TRGB = pd.read_csv('%s/Data_Anand2021/SN_TRGB.csv'%data_dir).replace(99999, np.nan)
    SN = TRGB[['Gal', 'SN', 'm_b', 'sig_m_b', 'A']].iloc[:-1, :]
    TRGB = TRGB.drop(columns=['SN', 'm_b', 'sig_m_b']).drop_duplicates().reset_index().drop(columns='index')
    galaxies = TRGB['Gal'].drop_duplicates().reset_index()['Gal']

    TRGB['z_obs'] = 1e-3 * np.array([2.41, 2.96, 0.8, 6.01, 5.45, 6.49, 3.90,\
                                    5.48, 1.46, 2.06, 6.03, 4.00, 1.49])  #  From NED

    return TRGB, SN, galaxies
