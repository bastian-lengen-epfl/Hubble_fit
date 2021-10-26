from Loading_data_functions import *
from Pipeline import *

##############################################################
###################### LOADING THE DATA ######################
##############################################################

# Dataframes for data
Cepheids, SN, SN_Pantheon, galaxies = load_data('./Data')

# Dataframe for the different simulations
Hubble = pd.DataFrame(columns=['Sim','q','H_0','chi2','cov','y','L', 'sig_H_0'])

# Dataframe for the outliers
Empty_Outliers = pd.DataFrame({'Gal':[],'logP':[],'m_W':[],'sig_m_W':[],'M/H':[],'pi':[],\
                                  'sig_pi':[], 'z_obs':[]})

# Do a save of the Cepheids with no rejection so we can reload it later
Save_Cepheids = Cepheids



##############################################################
######################## Simulations #########################
##############################################################

### Normal simulations -> added dispersion = 0
update_added_dispersion(0.0000)
Hubble, Cepheids, Cepheids_Outliers = multi_run(Hubble, Save_Cepheids, Empty_Outliers,SN, galaxies, 'Normal')

### Added dispersion -> added dispersion = 0.0682
update_added_dispersion(0.0682)
Hubble, Cepheids, Cepheids_Outliers = multi_run(Hubble, Save_Cepheids, Empty_Outliers,SN, galaxies, 'Added_Dispersion')

### Final Hubble plot
Hubble_plot(Hubble)