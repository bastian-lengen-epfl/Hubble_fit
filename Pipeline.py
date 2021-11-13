from Plot_functions import *
from Errors_functions import *
from Outliers_functions import *
from Relativistics_functions import *

def single_run_Cepheids(Cepheids: pd.DataFrame, Cepheids_Outliers: pd.DataFrame, SN: pd.DataFrame,\
               galaxies: list, name: str, fig_dir: str = "./Figure"):
    # Run the fit
    q, H_0, chi2, cov, y, L, sigma_H_0 = find_H0_Cepheids(Cepheids, SN, galaxies, name, display_text=True)

    # Plot PL relations:
    plot_PL(Cepheids, Cepheids_Outliers, galaxies, q, name, fig_dir)
    plot_global_PL(Cepheids, Cepheids_Outliers, galaxies, q, name, fig_dir)

    # Errors analysis
    errors = np.array(y - np.matmul(L, q))[0:len(Cepheids)]
    errors_distribution(errors, name)

    return [name, q, H_0, chi2, cov, y, L, sigma_H_0]

def multi_run_Cepheids(Hubble: pd. DataFrame, Cepheids: pd.DataFrame, Cepheids_Outliers: pd.DataFrame,\
              SN: pd.DataFrame, galaxies: list, name: str, fig_dir: str = "./Figure"):
    ######### no outliers, no relativistic corrections
    print('\n-------------------------------------\n%s\n-------------------------------------' % name)

    # Run the routine
    Hubble.loc[len(Hubble)] = single_run_Cepheids(Cepheids, Cepheids_Outliers, SN, galaxies, name, fig_dir)

    ######### outliers, no relativistic corrections

    # Run the rejection
    Cepheids, Cepheids_Outliers = kappa_clipping(kappa, Cepheids, Cepheids_Outliers, SN, galaxies, name, all_text=False)

    name = name + '_Out'
    print('\n-------------------------------------\n%s (%i outliers) \n-------------------------------------' % (
    name, len(Cepheids_Outliers)))

    # Run the routine
    Hubble.loc[len(Hubble)] = single_run_Cepheids(Cepheids, Cepheids_Outliers, SN, galaxies, name, fig_dir)

    ######### outliers, RLB, no K-correction

    # Correct the dataset (here logP) for the RLB
    Cepheids = Leavitt_correction(Cepheids, SN, galaxies)

    name = name + '_RLB'
    print('\n-------------------------------------\n%s (%i outliers) \n-------------------------------------' % (
    name, len(Cepheids_Outliers)))

    # Run the routine
    Hubble.loc[len(Hubble)] = single_run_Cepheids(Cepheids, Cepheids_Outliers, SN, galaxies, name, fig_dir)

    # Distance plot before/after corrections
    RLB_galaxies_distance(Hubble.loc[Hubble.index[-2], 'q'], \
                          Hubble.loc[Hubble.index[-1], 'q'], galaxies, name, fig_dir)

    ######### outliers, RLB,  K-correction
    name = name + '_Kcorr'
    print('\n-------------------------------------\n%s (%i outliers) \n-------------------------------------' % (
    name, len(Cepheids_Outliers)))

    # Correct the dataset (here m_W) for the k correction
    Cepheids = Kcorr(Cepheids)

    Hubble.loc[len(Hubble)] = single_run_Cepheids(Cepheids, Cepheids_Outliers, SN, galaxies, name, fig_dir)

    return Hubble, Cepheids, Cepheids_Outliers

def run_TRGB(Hubble: pd. DataFrame ,TRGB: pd.DataFrame, SN: pd.DataFrame, galaxies: list):
    # Run the fit
    print('\n-------------------------------------\nTRGB\n-------------------------------------')
    q, H_0, chi2, cov, y, L, sigma_H_0 = find_H0_TRGB(TRGB, SN, galaxies, display_text=True)
    Hubble.loc[len(Hubble)] = ['TRGB',q, H_0, chi2, cov, y, L, sigma_H_0]
    return Hubble


def single_run_Both(Cepheids: pd.DataFrame, Cepheids_Outliers: pd.DataFrame, SN_Cepheids: pd.DataFrame,\
                    galaxies_Cepheids: list, name: str, TRGB: pd.DataFrame, SN_TRGB: pd.DataFrame, galaxies_TRGB: list,\
                    fig_dir: str = "./Figure"):
    # Run the fit /!\ here H_0 and sigma_H_0 are vectors
    q, H_0, chi2, cov, y, L, sigma_H_0 = find_H0_Both(Cepheids, SN_Cepheids, galaxies_Cepheids, name,\
                                                      TRGB, SN_TRGB, galaxies_TRGB, display_text=True)

    # Plot PL relations:
    plot_PL(Cepheids, Cepheids_Outliers, galaxies_Cepheids, q[0:27], name, fig_dir)
    plot_global_PL(Cepheids, Cepheids_Outliers, galaxies_Cepheids, q[0:27], name, fig_dir)

    # Errors analysis
    errors = np.array(y - np.matmul(L, q))[0:len(Cepheids)]
    errors_distribution(errors, name)

    return [name+'_Cep', q, H_0[0], chi2, cov, y, L, sigma_H_0[0]], \
           [name+'_TRGB', q, H_0[1], chi2, cov, y, L, sigma_H_0[1]], \
           [name+'-BOTH', q, H_0[2], chi2, cov, y, L, sigma_H_0[2]],

def multi_run_Both(Hubble: pd. DataFrame, Cepheids: pd.DataFrame, Cepheids_Outliers: pd.DataFrame, \
                   SN_Cepheids: pd.DataFrame, galaxies_Cepheids: list, TRGB: pd.DataFrame, SN_TRGB: pd.DataFrame, \
                   galaxies_TRGB: list, name: str, fig_dir: str = "./Figure"):
    ######### no outliers, no added dispersion, no relativistic corrections
    print('\n-------------------------------------\n%s\n-------------------------------------' % name)

    # Run the routine
    H0 = single_run_Both(Cepheids, Cepheids_Outliers, SN_Cepheids, galaxies_Cepheids,\
                         name, TRGB, SN_TRGB, galaxies_TRGB, fig_dir)
    Hubble.loc[len(Hubble)] = H0[0]
    Hubble.loc[len(Hubble)] = H0[1]
    Hubble.loc[len(Hubble)] = H0[2]


    ######### outliers, no added dispersion, no relativistic corrections

    # Run the rejection
    Cepheids, Cepheids_Outliers = kappa_clipping_both(kappa, Cepheids, Cepheids_Outliers, SN_Cepheids, \
                                                      galaxies_Cepheids, TRGB, SN_TRGB, galaxies_TRGB, \
                                                      name, all_text = False)

    name = name + '_Out'
    print('\n-------------------------------------\n%s (%i outliers) \n-------------------------------------' % (
        name, len(Cepheids_Outliers)))

    # Run the routine
    H0 = single_run_Both(Cepheids, Cepheids_Outliers, SN_Cepheids, galaxies_Cepheids, \
                         name, TRGB, SN_TRGB, galaxies_TRGB, fig_dir)
    Hubble.loc[len(Hubble)] = H0[0]
    Hubble.loc[len(Hubble)] = H0[1]
    Hubble.loc[len(Hubble)] = H0[2]

    ######### outliers, RLB, no added dispersion, no K-correction

    # Correct the dataset (here logP) for the RLB
    Cepheids = Leavitt_correction(Cepheids, SN_Cepheids, galaxies_Cepheids)

    name = name + '_RLB'
    print('\n-------------------------------------\n%s (%i outliers) \n-------------------------------------' % (
        name, len(Cepheids_Outliers)))

    # Run the routine
    H0 = single_run_Both(Cepheids, Cepheids_Outliers, SN_Cepheids, galaxies_Cepheids, \
                         name, TRGB, SN_TRGB, galaxies_TRGB, fig_dir)
    Hubble.loc[len(Hubble)] = H0[0]
    Hubble.loc[len(Hubble)] = H0[1]
    Hubble.loc[len(Hubble)] = H0[2]

    # Distance plot before/after corrections
    RLB_galaxies_distance(Hubble.loc[Hubble.index[-2], 'q'], \
                          Hubble.loc[Hubble.index[-1], 'q'], galaxies_Cepheids, name, fig_dir)

    ######### outliers, RLB,  K-correction, no added dispersion
    name = name + '_Kcorr'
    print('\n-------------------------------------\n%s (%i outliers) \n-------------------------------------' % (
        name, len(Cepheids_Outliers)))

    # Correct the dataset (here m_W) for the k correction
    Cepheids = Kcorr(Cepheids)

    H0 = single_run_Both(Cepheids, Cepheids_Outliers, SN_Cepheids, galaxies_Cepheids, \
                         name, TRGB, SN_TRGB, galaxies_TRGB, fig_dir)
    Hubble.loc[len(Hubble)] = H0[0]
    Hubble.loc[len(Hubble)] = H0[1]
    Hubble.loc[len(Hubble)] = H0[2]

    return Hubble, Cepheids, Cepheids_Outliers