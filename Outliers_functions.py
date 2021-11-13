from Fit_functions import *

def kappa_clipping(kappa: float, Cepheids: pd.DataFrame, Cepheids_Outliers: pd.DataFrame, SN: pd.DataFrame ,\
                   galaxies: list, name: str, all_text: bool = False):
    # First iteration
    q, H_0, chi2, cov, y, L, sigma_H_0 = find_H0_Cepheids(Cepheids, SN, galaxies, name, all_text)

    # Initialize the algorithm
    errors = np.array(y - np.matmul(L, q))[0:len(Cepheids)]
    std = np.std(errors)
    threshold = np.sum(Cepheids['Gal'] == 'MW') # to correct y order to Cepheids order
    errors = np.append(errors[threshold:], errors[:threshold])  #  Reoder errors according to Cepheids order
    worst = np.argmax(np.abs(errors))

    # Start iterations
    i = 0
    while np.abs(errors[worst]) >= kappa * std:
        # Reject worst outlier :
        Cepheids_Outliers = Cepheids_Outliers.append(Cepheids.iloc[worst])
        Cepheids = Cepheids.drop(worst)
        Cepheids.index = range(0, len(Cepheids)) # Re-index
        Cepheids_Outliers.index = range(0, len(Cepheids_Outliers))

        # Can display details if wanted
        # print("\nDrop %f from Cepheids from %s\n" % (worst, Cepheids_Outliers['Gal'].iloc[-1]))
        # print("Errors = %f, Clipping threshold = %f" % (errors[worst], kappa*std))

        # re-iterate
        q, H_0, chi2, cov, y, L, sigma_H_0 = find_H0_Cepheids(Cepheids, SN, galaxies, name, all_text)

        errors = np.array(y - np.matmul(L, q))[0:len(Cepheids)]
        std = np.std(errors)
        threshold = np.sum(Cepheids['Gal'] == 'MW') # to correct y order to Cepheids order
        errors = np.append(errors[threshold:], errors[:threshold])  #  Reoder errors according to Cepheids order
        worst = np.argmax(np.abs(errors))

    return Cepheids, Cepheids_Outliers

def kappa_clipping_both(kappa: float, Cepheids: pd.DataFrame, Cepheids_Outliers: pd.DataFrame, \
        SN_Cepheids: pd.DataFrame, galaxies_Cepheids: list, TRGB: pd.DataFrame, SN_TRGB: pd.DataFrame, \
        galaxies_TRGB: list, name: str, all_text: bool = False):

    # First iteration
    q, H_0, chi2, cov, y, L, sigma_H_0 = find_H0_Both(Cepheids, SN_Cepheids, galaxies_Cepheids, name, \
                                                      TRGB, SN_TRGB, galaxies_TRGB, all_text)

    # Initialize the algorithm
    errors = np.array(y - np.matmul(L, q))[0:len(Cepheids)]
    std = np.std(errors)
    threshold = np.sum(Cepheids['Gal'] == 'MW') # to correct y order to Cepheids order
    errors = np.append(errors[threshold:], errors[:threshold])  #  Reoder errors according to Cepheids order
    worst = np.argmax(np.abs(errors))

    # Start iterations
    while np.abs(errors[worst]) >= kappa * std:
        # Reject worst outlier :
        Cepheids_Outliers = Cepheids_Outliers.append(Cepheids.iloc[worst])
        Cepheids = Cepheids.drop(worst)
        Cepheids.index = range(0, len(Cepheids)) # Re-index
        Cepheids_Outliers.index = range(0, len(Cepheids_Outliers))

        # Can display details if wanted
        # print("\nDrop %f from Cepheids from %s\n" % (worst, Cepheids_Outliers['Gal'].iloc[-1]))
        # print("Errors = %f, Clipping threshold = %f" % (errors[worst], kappa*std))

        # re-iterate
        q, H_0, chi2, cov, y, L, sigma_H_0 = find_H0_Both(Cepheids, SN_Cepheids, galaxies_Cepheids, name, \
                                                          TRGB, SN_TRGB, galaxies_TRGB, all_text)

        errors = np.array(y - np.matmul(L, q))[0:len(Cepheids)]
        std = np.std(errors)
        threshold = np.sum(Cepheids['Gal'] == 'MW') # to correct y order to Cepheids order
        errors = np.append(errors[threshold:], errors[:threshold])  #  Reoder errors according to Cepheids order
        worst = np.argmax(np.abs(errors))

    return Cepheids, Cepheids_Outliers
