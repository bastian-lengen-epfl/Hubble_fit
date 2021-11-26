###  Fit functions
import numpy as np
import pandas as pd
from Values import *
from Usefull_functions import redshift_magnitude_x


# Fit the SN_pantheon for a_b,  with z_min and z_max the limit for the fit
def find_a_b(SN: pd.DataFrame, z_min: float = 0.023, z_max: float = 0.15, display_text: bool = True):
    # Rename the vector
    z = SN['zcmb']
    dz = SN['dz']
    m = SN['mb']
    dm = SN['dmb']

    # Fit boundaries
    z_fit = z[(z < z_max) & (z > z_min)]
    m_fit = m[(z < z_max) & (z > z_min)]
    dz_fit = dz[(z < z_max) & (z > z_min)]
    dm_fit = dm[(z < z_max) & (z > z_min)]

    m = m[(z >= z_max) | (z <= z_min)]
    dz = dz[(z >= z_max) | (z <= z_min)]
    dm = dm[(z >= z_max) | (z <= z_min)]
    z = z[(z >= z_max) | (z <= z_min)]

    # Linear regression
    a_b = np.mean(redshift_magnitude_x(z_fit) - 0.2 * m_fit)
    da_b = 0

    # Compute the error
    errors = (a_b + 0.2 * m_fit) - redshift_magnitude_x(z_fit)
    std = np.std(errors)

    # Reject outliers with errors > 2.7*std (and put them in the unfitted vectors z,dz,m,dm)
    kappa = 2.7
    z = np.append(z, z_fit[errors >= kappa * std])
    dz = np.append(dz, dz_fit[errors >= kappa * std])
    m = np.append(m, m_fit[errors >= kappa * std])
    dm = np.append(dm, dm_fit[errors >= kappa * std])
    z_fit = z_fit[errors < kappa * std]
    m_fit = m_fit[errors < kappa * std]
    dz_fit = dz_fit[errors < kappa * std]
    dm_fit = dm_fit[errors < kappa * std]

    # Linear regression
    a_b = np.mean(redshift_magnitude_x(z_fit) - 0.2 * m_fit)
    da_b = np.mean(dz_fit ** 2 + 0.2 * dm_fit) / np.sqrt(len(z_fit))
    # Print
    if display_text == True:
        print('a_b = %f+/-%f' % (a_b, da_b))

    return a_b, [z, dz, m, dm, z_fit, dz_fit, m_fit, dm_fit]

# Fit H0 for Cepheids
def find_H0_Cepheids(Cepheids: pd.DataFrame, SN: pd.DataFrame , galaxies: list, name:  str, display_text: bool =False):  ### Moertsell 2021
    ### Create y vector
    MW_filter = Cepheids['Gal'] == 'MW'
    y = np.array(Cepheids[MW_filter]['m_W'] - 10 + 5 * np.log10(Cepheids[MW_filter]['pi']))
    y = np.append(y, Cepheids[~MW_filter]['m_W'])
    y = np.append(y, [mu_N4258, mu_LMC])
    y = np.append(y, SN['m_b'])

    ### Create the parameters vector
    # 0-18 : mu_host // 19-20 : mu_N4258, mu_LMC // 21 : mW
    #  22-23 : bs, bl // 24-25 : ZW, zp // 26 : MB
    q = np.zeros(27)

    ### Create vector P and [O/H] in the good galaxy order
    logP = np.append(Cepheids[MW_filter]['logP'], Cepheids[~MW_filter]['logP'])
    MH = np.append(Cepheids[MW_filter]['M/H'], Cepheids[~MW_filter]['M/H'])

    ### Create the Design matrix :
    L = np.zeros((len(y), len(q)))
    Cepheids_count = 0
    SN_count = 0
    for i in range(0, len(y)):
        if i < len(Cepheids[MW_filter]):
            MW_index_offset = len(Cepheids[~MW_filter])  #  since MW are last in DF and here first.
            L[i][21] = 1  #  mW
            if logP[i] < 1:
                L[i][22] = logP[i] - 1 # bs
            else:
                L[i][23] = logP[i] - 1 # bl
            L[i][24] = MH[i] # ZW
            L[i][25] = -5 / np.log(10) / Cepheids.loc[MW_index_offset + i, 'pi'] # zp
        elif i < len(Cepheids):
            galaxy_index = np.where(galaxies == Cepheids.loc[Cepheids_count, 'Gal'])[0][0] #  Index restart at 0
            Cepheids_count = Cepheids_count + 1
            L[i][galaxy_index] = 1  # mu_host
            L[i][21] = 1  #  mW
            if logP[i] < 1:
                L[i][22] = logP[i] - 1 # bs
            else:
                L[i][23] = logP[i] - 1 # bl
            L[i][24] = MH[i] # ZW
        elif i < len(Cepheids) + 1:
            L[i][19] = 1 # mu_N4258
        elif i < len(Cepheids) + 2:
            L[i][20] = 1 # mu_LMC
        else:
            L[i][SN_count] = 1 # mu_host
            SN_count = SN_count + 1
            L[i][26] = 1 # MB

    ### Create the correlation matrix :
    if name[0] == 'N':
        extra_scatter = 0.0 # extra scatter for the hosts cepheids
    else :
        extra_scatter = added_scatter
    sigma2_pi = (Cepheids[MW_filter]['sig_m_W'] ** 2 + extra_scatter**2) \
                + (5 / np.log(10) / Cepheids[MW_filter]['pi'] * Cepheids[MW_filter]['sig_pi']) ** 2
    diag_sigma = np.array(sigma2_pi)  # for MW
    diag_sigma = np.append(diag_sigma,
                           Cepheids[~MW_filter]['sig_m_W'] ** 2 + extra_scatter ** 2)  # for host, N4258 & LMC
    diag_sigma = np.append(diag_sigma, [sigma_N4258 ** 2, sigma_LMC ** 2])  # geometric distances
    diag_sigma = np.append(diag_sigma, SN['sig'] ** 2)
    C = np.diag(diag_sigma)

    ### Find the optimal parameters :
    LT = np.transpose(L)
    C1 = np.linalg.inv(C)
    cov = np.linalg.inv(np.matmul(np.matmul(LT, C1), L))
    q = np.matmul(np.matmul(np.matmul(cov, LT), C1), y)
    chi2 = np.matmul(np.matmul(np.transpose(y - np.matmul(L, q)), C1), y - np.matmul(L, q))

    mu_hat_N4258, mu_hat_LMC, m_w_H, b_s, b_l, Z_W, zp, M_B = q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26]
    logH_0 = ((M_B + 5 * a_b + 25) / 5)
    H_0 = 10 ** logH_0
    sigma_M_B = np.sqrt(cov[26][26])
    sigma_logH0 = np.sqrt((0.2 * sigma_M_B) ** 2 + sigma_a_b ** 2)
    sigma_H_0 = np.log(10) * H_0 * sigma_logH0
    sigma_mu_hat_N4258, sigma_mu_hat_LMC = np.sqrt(cov[19][19]), np.sqrt(cov[20][20])
    sigma_m_w_H, sigma_b_s, sigma_b_l = np.sqrt(cov[21][21]), np.sqrt(cov[22][22]), np.sqrt(cov[23][23])
    sigma_Z_W, sigma_zp = np.sqrt(cov[24][24]), np.sqrt(cov[25][25])

    if display_text == True :
        print('mu_hat_N4258 = %f +/- %f' % (mu_hat_N4258, sigma_mu_hat_N4258))
        print('mu_hat_LMC = %f +/- %f' % (mu_hat_LMC, sigma_mu_hat_LMC))
        print('m_w_H = %f +/- %f' % (m_w_H, sigma_m_w_H))
        print('b_s = %f +/- %f' % (b_s, sigma_b_s))
        print('b_l = %f +/- %f' % (b_l, sigma_b_l))
        print('Z_W = %f +/- %f' % (Z_W, sigma_Z_W))
        print('zp = %f +/- %f' % (zp, sigma_zp))
        print('M_B = %f +/- %f' % (M_B, sigma_M_B))
        print('a_b = %f +/- %f' % (a_b, sigma_a_b))
        print('H_0 = %f +/- %f' % (H_0, sigma_H_0))
        print('chi2 = %f' % chi2)
        dof = len(y)-len(q)
        print('chi2/dof = %f' % (chi2 / dof))

    return q, H_0, chi2, cov, y, L, sigma_H_0

# Fit TRGB with free M_TRGB and slope a
def find_H0_TRGB(TRGB: pd.DataFrame, SN: pd.DataFrame , galaxies: list, display_text: bool = False):
    ### Create y vector
    y = np.array(TRGB['m'] - TRGB['A']-0.2*(TRGB['V-I']-TRGB.loc[len(TRGB)-1, 'V-I']))  # TRGB magnitude V-I
    y = np.append(y, mu_N4258)
    y = np.append(y, SN['m_b'])  # SN magnitude

    ### Create parameters vector :
    # 0-11: mu_host, 12: mu_N4258, 13: M_TRGB, 14: M_B
    q = np.zeros(15)

    ### Create the design matrix
    L = np.zeros((len(y), len(q)))
    TRGB_count = 0
    SN_count = 0
    for i in range(0, len(y)):
        if i < len(TRGB):
            L[i][TRGB_count] = 1  # mu
            L[i][13] = 1  #  M_TRGB
            TRGB_count = TRGB_count + 1
        elif i < len(TRGB) + 1:
            L[i][12] = 1  # mu_N4258
        else:
            galaxy_index = np.where(galaxies == SN.loc[SN_count, 'Gal'])[0][0]
            L[i][galaxy_index] = 1  # mu
            L[i][14] = 1  # M_B
            SN_count = SN_count + 1

    ### Create the correlation matrix
    diag_sigma = np.array(TRGB['sig_m'] ** 2)  #  TRGB sigma
    diag_sigma = np.append(diag_sigma, sigma_N4258 ** 2)  # geom distance
    diag_sigma = np.append(diag_sigma, SN['sig_m_b'] ** 2)  # SN sigma
    C = np.diag(diag_sigma)

    ### Find the optimal parameters :
    LT = np.transpose(L)
    C1 = np.linalg.inv(C)
    cov = np.linalg.inv(np.matmul(np.matmul(LT, C1), L))
    q = np.matmul(np.matmul(np.matmul(cov, LT), C1), y)
    chi2 = np.matmul(np.matmul(np.transpose(y - np.matmul(L, q)), C1), y - np.matmul(L, q))

    mu_hat_N4258, M_TRGB, M_B = q[12], q[13], q[14]
    logH_0 = ((M_B + 5 * a_b + 25) / 5)
    H_0 = 10 ** logH_0

    sigma_M_B = np.sqrt(cov[14][14])
    sigma_logH0 = np.sqrt((0.2 * sigma_M_B) ** 2 + sigma_a_b ** 2)
    sigma_H_0 = np.log(10) * H_0 * sigma_logH0
    sigma_mu_hat_N4258 = np.sqrt(cov[12][12])
    sigma_M_TRGB = np.sqrt(cov[13][13])

    if display_text == True:
        print('mu_hat_N4258 = %f +/- %f' % (mu_hat_N4258, sigma_mu_hat_N4258))
        print('M_TRGB = %f +/- %f' % (M_TRGB, sigma_M_TRGB))
        print('M_B = %f +/- %f' % (M_B, sigma_M_B))
        print('a_b = %f +/- %f' % (a_b, sigma_a_b))
        print('H_0 = %f +/- %f' % (H_0, sigma_H_0))
        print('chi2 = %f' % chi2)
        dof = len(y) - len(q)
        print('chi2/dof = %f' % (chi2 / dof))

    return q, H_0, chi2, cov, y, L, sigma_H_0

# Fit Cepheids and TRGB at the same time
def find_H0_Both(Cepheids: pd.DataFrame, SN_Cepheids: pd.DataFrame, galaxies_Cepheids: list, name:  str,\
                 TRGB: pd.DataFrame, SN_TRGB: pd.DataFrame, galaxies_TRGB: list, display_text: bool = True):
    ### Galaxy in both DF and only in TRGB
    galaxies_Both = np.array([])
    galaxies_Noth = np.array([])

    for galaxy in galaxies_TRGB:
        if galaxy in list(galaxies_Cepheids):
            galaxies_Both = np.append(galaxies_Both, galaxy)
        else:
            galaxies_Noth = np.append(galaxies_Noth, galaxy)

    ### Create y vector
    # Cepheids
    MW_filter = Cepheids['Gal'] == 'MW'
    y = np.array(Cepheids[MW_filter]['m_W'] - 10 + 5 * np.log10(Cepheids[MW_filter]['pi']))
    y = np.append(y, Cepheids[~MW_filter]['m_W'])
    y = np.append(y, [mu_N4258, mu_LMC])
    y = np.append(y, SN_Cepheids['m_b'])
    # TRGB
    y = np.append(y, TRGB['m'] - TRGB['A']-0.2*(TRGB['V-I']-TRGB.loc[len(TRGB)-1, 'V-I']))  # TRGB magnitude V-I
    y = np.append(y, SN_TRGB['m_b'])  # SN magnitude

    ### Create the parameters vector
    # Cepheids
    # 0-18 : mu_host // 19-20 : mu_N4258, mu_LMC // 21 : mW
    # 22-23 : bs, bl // 24-25 : ZW, zp // 26 : MB
    # TRGB
    # 27-32 : mu_Noth // 33-38 : delta_mu_Both # One less because no delta_mu_N4258
    # 39 : M_TRGB // 40 : delta_MB
    q = np.zeros(41)

    ### Create vector P and [O/H] in the good galaxy order for the Cepheids
    logP = np.append(Cepheids[MW_filter]['logP'], Cepheids[~MW_filter]['logP'])
    MH = np.append(Cepheids[MW_filter]['M/H'], Cepheids[~MW_filter]['M/H'])

    ### Create the design matrix
    L = np.zeros((len(y), len(q)))
    Cepheids_count, TRGB_count, SNc_count, SNt_count = 0,0,0,0
    for i in range(0, len(y)):
        # Cepheids
        if i < len(Cepheids[MW_filter]):
            MW_index_offset = len(Cepheids[~MW_filter])  #  since MW are last in DF and here first.
            L[i][21] = 1  #  mW
            if logP[i] < 1:
                L[i][22] = logP[i] - 1  # bs
            else:
                L[i][23] = logP[i] - 1  # bl
            L[i][24] = MH[i]  # ZW
            L[i][25] = -5 / np.log(10) / Cepheids.loc[MW_index_offset + i, 'pi']  # zp
        elif i < len(Cepheids):
            galaxy_index = np.where(galaxies_Cepheids == Cepheids.loc[Cepheids_count, 'Gal'])[0][0]  #  Index restart at 0
            Cepheids_count = Cepheids_count + 1
            L[i][galaxy_index] = 1  # mu_host
            L[i][21] = 1  #  mW
            if logP[i] < 1:
                L[i][22] = logP[i] - 1  # bs
            else:
                L[i][23] = logP[i] - 1  # bl
            L[i][24] = MH[i]  # ZW
        elif i < len(Cepheids) + 1:
            L[i][19] = 1  # mu_N4258
        elif i < len(Cepheids) + 2:
            L[i][20] = 1  # mu_LMC
        elif i< len(Cepheids) + 2 + len(SN_Cepheids):
            L[i][SNc_count] = 1  # mu_host
            SNc_count = SNc_count + 1
            L[i][26] = 1  # MB

        # TRGB
        elif i< len(Cepheids) + 2 + len(SN_Cepheids) + len(TRGB):
            if TRGB.loc[TRGB_count, 'Gal'] in list(galaxies_Both):
                galaxy_index = np.where(galaxies_Cepheids == TRGB.loc[TRGB_count, 'Gal'])[0][0]
                if TRGB.loc[TRGB_count, 'Gal'] != 'N4258':
                    delta_index = np.where(galaxies_Both == TRGB.loc[TRGB_count, 'Gal'])[0][0]
                    L[i][delta_index+27+len(galaxies_Noth)] = 1 # delta_mu_host
            else :
                galaxy_index = 27+np.where(galaxies_Noth == TRGB.loc[TRGB_count, 'Gal'])[0][0]
            L[i][galaxy_index] = 1  # mu_host
            L[i][39] = 1  #  M_TRGB
            TRGB_count = TRGB_count + 1

        # SN TRGB
        else :
            if SN_TRGB.loc[SNt_count, 'Gal'] in list(galaxies_Both):
                galaxy_index = np.where(galaxies_Cepheids == SN_TRGB.loc[SNt_count, 'Gal'])[0][0]
                if SN_TRGB.loc[SNt_count, 'Gal'] != 'N4258':
                    delta_index = np.where(galaxies_Both == SN_TRGB.loc[TRGB_count, 'Gal'])[0][0]
                    L[i][delta_index+27+len(galaxies_Noth)] = 1 # delta_mu_host
            else :
                galaxy_index = 27+np.where(galaxies_Noth == SN_TRGB.loc[SNt_count, 'Gal'])[0][0]
            L[i][galaxy_index] = 1  # mu_host
            L[i][26] = 1 # M_B
            L[i][40] = 1 # Delta_MB
            SNt_count = SNt_count + 1

    ### Create the correlation matrix :
    if name[0] == 'N':
        extra_scatter = 0.0  # extra scatter for the hosts cepheids
    else:
        extra_scatter = added_scatter
    sigma2_pi = (Cepheids[MW_filter]['sig_m_W'] ** 2 + extra_scatter**2)\
                + (5 / np.log(10) / Cepheids[MW_filter]['pi'] * Cepheids[MW_filter]['sig_pi']) ** 2
    diag_sigma = np.array(sigma2_pi)  # for MW
    diag_sigma = np.append(diag_sigma,
                           Cepheids[~MW_filter]['sig_m_W'] ** 2 + extra_scatter ** 2)  # for host, N4258 & LMC
    diag_sigma = np.append(diag_sigma, [sigma_N4258 ** 2, sigma_LMC ** 2])  # geometric distances
    diag_sigma = np.append(diag_sigma, SN_Cepheids['sig'] ** 2)
    diag_sigma = np.append(diag_sigma, TRGB['sig_m'] ** 2)  #  TRGB sigma
    diag_sigma = np.append(diag_sigma, SN_TRGB['sig_m_b'] ** 2)  # SN sigma
    C = np.diag(diag_sigma)

    ### Find the optimal parameters :
    LT = np.transpose(L)
    C1 = np.linalg.inv(C)
    cov = np.linalg.inv(np.matmul(np.matmul(LT, C1), L))
    q = np.matmul(np.matmul(np.matmul(cov, LT), C1), y)
    chi2 = np.matmul(np.matmul(np.transpose(y - np.matmul(L, q)), C1), y - np.matmul(L, q))

    # Cepheids
    mu_hat_N4258, mu_hat_LMC, m_w_H, b_s, b_l, Z_W, zp, M_B_Cep = q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26]
    logH_0_Cep = ((M_B_Cep + 5 * a_b + 25) / 5)
    H_0_Cep = 10 ** logH_0_Cep
    sigma_M_B_Cep = np.sqrt(cov[26][26])
    sigma_logH0_Cep = np.sqrt((0.2 * sigma_M_B_Cep) ** 2 + sigma_a_b ** 2)
    sigma_H_0_Cep = np.log(10) * H_0_Cep * sigma_logH0_Cep
    sigma_mu_hat_N4258, sigma_mu_hat_LMC = np.sqrt(cov[19][19]), np.sqrt(cov[20][20])
    sigma_m_w_H, sigma_b_s, sigma_b_l = np.sqrt(cov[21][21]), np.sqrt(cov[22][22]), np.sqrt(cov[23][23])
    sigma_Z_W, sigma_zp = np.sqrt(cov[24][24]), np.sqrt(cov[25][25])

    # TRGB
    M_TRGB, delta_M_B = q[39], q[40]
    M_B_TRGB = M_B_Cep+delta_M_B
    logH_0_TRGB = ((M_B_TRGB + 5 * a_b + 25) / 5)
    H_0_TRGB = 10 ** logH_0_TRGB

    sigma_delta_M_B = np.sqrt(cov[40][40])
    sigma_M_B_TRGB = np.sqrt(sigma_M_B_Cep**2+sigma_delta_M_B**2)/np.sqrt(2)
    sigma_logH0_TRGB = np.sqrt((0.2 * sigma_M_B_TRGB) ** 2 + sigma_a_b ** 2)
    sigma_H_0_TRGB = np.log(10) * H_0_TRGB * sigma_logH0_TRGB
    sigma_M_TRGB = np.sqrt(cov[39][39])

    # AVG
    weights = np.array([1/sigma_H_0_Cep, 1/sigma_H_0_TRGB])/np.sum([1/sigma_H_0_Cep, 1/sigma_H_0_TRGB])
    H_0_AVG = np.dot(weights, [H_0_TRGB, H_0_Cep])
    sigma_H_0_AVG = np.dot(weights, [sigma_H_0_TRGB, sigma_H_0_Cep])/np.sqrt(2)


    if display_text == True :
        print('\nCEPHEIDS :\n')
        print('m_w_H = %f +/- %f' % (m_w_H, sigma_m_w_H))
        print('b_s = %f +/- %f' % (b_s, sigma_b_s))
        print('b_l = %f +/- %f' % (b_l, sigma_b_l))
        print('Z_W = %f +/- %f' % (Z_W, sigma_Z_W))
        print('zp = %f +/- %f' % (zp, sigma_zp))
        print('M_B = %f +/- %f' % (M_B_Cep, sigma_M_B_Cep))
        print('H_0 = %f +/- %f' % (H_0_Cep, sigma_H_0_Cep))

        print('\nTRGB :\n')
        print('M_TRGB = %f +/- %f' % (M_TRGB, sigma_M_TRGB))
        print('M_B = %f +/- %f' % (M_B_TRGB, sigma_M_B_TRGB))
        print('H_0 = %f +/- %f' % (H_0_TRGB, sigma_H_0_TRGB))

        print('\nBOTH :\n')
        print('mu_hat_N4258 = %f +/- %f' % (mu_hat_N4258, sigma_mu_hat_N4258))
        print('mu_hat_LMC = %f +/- %f' % (mu_hat_LMC, sigma_mu_hat_LMC))
        print('Delta_M_B = %f+/-%f'%(delta_M_B , sigma_delta_M_B))
        print('a_b = %f +/- %f' % (a_b, sigma_a_b))
        print('chi2 = %f' % chi2)
        dof = len(y)-len(q)
        print('chi2/dof = %f' % (chi2 / dof))
        print('H_0 = %f +/- %f'%(H_0_AVG, sigma_H_0_AVG))

    return q, [H_0_Cep, H_0_TRGB, H_0_AVG], chi2, cov, y, L, [sigma_H_0_Cep, sigma_H_0_TRGB, sigma_H_0_AVG]