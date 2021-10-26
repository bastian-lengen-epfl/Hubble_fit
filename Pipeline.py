from Plot_functions import *
from Errors_functions import *
from Outliers_functions import *
from Relativistics_functions import *

def single_run(Cepheids: pd.DataFrame, Cepheids_Outliers: pd.DataFrame, SN: pd.DataFrame,\
               galaxies: list, name: str, fig_dir: str = "./Figure"):
    # Run the fit
    q, H_0, chi2, cov, y, L, sigma_H_0 = find_H0(Cepheids, SN, galaxies, name, display_text=True)

    # Plot PL relations:
    plot_PL(Cepheids, Cepheids_Outliers, galaxies, q, name, fig_dir)
    plot_global_PL(Cepheids, Cepheids_Outliers, galaxies, q, name, fig_dir)

    # Errors analysis
    errors = np.array(y - np.matmul(L, q))[0:len(Cepheids)]
    errors_distribution(errors, name)

    return [name, q, H_0, chi2, cov, y, L, sigma_H_0]

def multi_run(Hubble: pd. DataFrame, Cepheids: pd.DataFrame, Cepheids_Outliers: pd.DataFrame,\
              SN: pd.DataFrame, galaxies: list, name: str, fig_dir: str = "./Figure"):
    ######### no outliers, no added dispersion, no relativistic corrections
    print('\n-------------------------------------\n%s\n-------------------------------------' % name)

    # Run the routine
    Hubble.loc[len(Hubble)] = single_run(Cepheids, Cepheids_Outliers, SN, galaxies, name, fig_dir)

    ######### outliers, no added dispersion, no relativistic corrections

    # Run the rejection
    Cepheids, Cepheids_Outliers = kappa_clipping(kappa, Cepheids, Cepheids_Outliers, SN, galaxies, name, all_text=False)

    name = name + '_Out'
    print('\n-------------------------------------\n%s (%i outliers) \n-------------------------------------' % (
    name, len(Cepheids_Outliers)))

    # Run the routine
    Hubble.loc[len(Hubble)] = single_run(Cepheids, Cepheids_Outliers, SN, galaxies, name, fig_dir)

    ######### outliers, RLB, no added dispersion, no K-correction

    # Correct the dataset (here logP) for the RLB
    Cepheids = Leavitt_correction(Cepheids, SN, galaxies)

    name = name + '_RLB'
    print('\n-------------------------------------\n%s (%i outliers) \n-------------------------------------' % (
    name, len(Cepheids_Outliers)))

    # Run the routine
    Hubble.loc[len(Hubble)] = single_run(Cepheids, Cepheids_Outliers, SN, galaxies, name, fig_dir)

    # Distance plot before/after corrections
    RLB_galaxies_distance(Hubble.loc[Hubble.index[-2], 'q'], \
                          Hubble.loc[Hubble.index[-1], 'q'], galaxies, name, fig_dir)

    ######### outliers, RLB,  K-correction, no added dispersion
    name = name + '_Kcorr'
    print('\n-------------------------------------\n%s (%i outliers) \n-------------------------------------' % (
    name, len(Cepheids_Outliers)))

    # Correct the dataset (here m_W) for the k correction
    Cepheids = Kcorr(Cepheids)

    Hubble.loc[len(Hubble)] = single_run(Cepheids, Cepheids_Outliers, SN, galaxies, name, fig_dir)

    return Hubble, Cepheids, Cepheids_Outliers