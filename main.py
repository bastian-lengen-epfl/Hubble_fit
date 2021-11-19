from Loading_data_functions import *
from Pipeline import *

##############################################################
###################### LOADING THE DATA ######################
##############################################################

# Dataframes for cepheids, TRGB, pantheon data
Cepheids, SN_Cepheids, galaxies_Cepheids = load_Cepheids('./Data')
TRGB, SN_TRGB, galaxies_TRGB = load_TRGB('./Data')
SN_Pantheon = load_SN_pantheon('./Data')

# Dataframe for the different simulations
Hubble = pd.DataFrame(columns=['Sim','q','H_0','chi2','cov','y','L', 'sig_H_0','C/T/Avg'])

# Dataframe for the outliers
Empty_Outliers = pd.DataFrame({'Gal':[],'logP':[],'m_W':[],'sig_m_W':[],'M/H':[],'pi':[],\
                                  'sig_pi':[], 'z_obs':[]})

# Do a save of the Cepheids with no rejection so we can reload it later
Save_Cepheids = Cepheids

##############################################################
###################### SN Pantheon ###########################
##############################################################
print("Intercept value from the pantheon sample : ")
z_min, z_max = 0.023, 0.15
a_b, fit_values = find_a_b(SN_Pantheon, z_min, z_max, True)
plot_SN(a_b, fit_values, z_min, z_max, './Figure')

##############################################################
######################## CEPHEIDS  ###########################
##############################################################

### Normal simulations -> added scatter = 0
Hubble, _, _ = multi_run_Cepheids(Hubble, Save_Cepheids, Empty_Outliers, \
                                                         SN_Cepheids, galaxies_Cepheids, 'N')

### Added dispersion -> added dispersion = 0.0682
Hubble, _, _ = multi_run_Cepheids(Hubble, Save_Cepheids, Empty_Outliers, \
                                                         SN_Cepheids, galaxies_Cepheids, 'D')


##############################################################
########################## TRGB  #############################
##############################################################

### Just need a single run since there is no outliers rejection
Hubble = run_TRGB(Hubble,TRGB,SN_TRGB,galaxies_TRGB)

### K correction
TRGB_Kcorr = Kcorr_TRGB(TRGB)
Hubble = run_TRGB(Hubble, TRGB_Kcorr, SN_TRGB, galaxies_TRGB, Kcorr=True)


##############################################################
#################### CEPHEIDS + TRGB  ########################
##############################################################

### Normal simulations -> added scatter = 0
Hubble, _, _ = multi_run_Both(Hubble, Save_Cepheids, Empty_Outliers, SN_Cepheids, \
                                                    galaxies_Cepheids, TRGB, SN_TRGB, galaxies_TRGB, 'N_Both')


### Added dispersion -> added dispersion = 0.0682
Hubble, _, _ = multi_run_Both(Hubble, Save_Cepheids, Empty_Outliers, SN_Cepheids, \
                                                    galaxies_Cepheids, TRGB, SN_TRGB, galaxies_TRGB, 'D_Both')


##############################################################
######################## Final DF  ###########################
##############################################################

### Final Hubble plot
Hubble_plot(Hubble)