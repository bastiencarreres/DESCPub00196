import sys
import os
import yaml
import time
import pandas as pd
import numpy as np
from pathlib import Path
import paper_fun as pf
import flip
from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)

zrange = [0.02, 0.1]
mock = int(sys.argv[1])
model = sys.argv[2]

print(f"PROCESSIGN MOCK {mock:02d}\n\n")
OUTPUT_DIR = f'/global/homes/b/bastienc/MY_SNANA_DIR/LSST_SNANA/PaperBBCVpec/results_{model.lower()}_sigu_alt/'

######################
# SET POWER SPECTRUM #
######################

kmin = (2 * np.pi) / 2000 # 2pi / L
kmax = 0.2

kh, ptt = pf.init_PS(1e-5, kmax)


############
# SET PATH #
############

PIPPIN_DIR = Path(os.environ['PIPPIN_OUTPUT'])
MOCK_DIR = PIPPIN_DIR / f'LSST_UCHUU_MOCK{mock:02d}_BC'
BBC_DIR = MOCK_DIR / f'6_BIASCOR/LSST_{model.upper()}/output'
FIT_DIR =  MOCK_DIR / '2_LCFIT'


with open(BBC_DIR / 'SUBMIT.INFO', 'r') as stream:
    data_loaded = yaml.safe_load(stream)

    
############
# INIT FIT #
############

RES = pf.init_res_dic(fittypes=['BBC', 'TRUE', 'STDFIT'], add_keys=['sigu'])


BBC_FILE = BBC_DIR / f'OUTPUT_BBCFIT/FITOPT000_MUOPT000.FITRES.gz'
FIT_FILE = FIT_DIR /  f'LSST_FIT_LSST_{model.upper()}/output/PIP_LSST_UCHUU_MOCK{mock:02d}_BC_LSST_{model.upper()}/FITOPT000.FITRES.gz'


#############
# LOAD DATA #
#############

# BBC DATA
df_BBC = ascii.read(BBC_FILE).to_pandas().set_index('CIDint')

# STANDARD DATA
df_STDFIT = ascii.read(FIT_FILE).to_pandas().set_index('CID')
df_STDFIT = df_STDFIT.loc[df_BBC.index]

# MASK
mask = ((df_BBC["HOST_NMATCH"] > 0) & (df_BBC['zHD'].between(zrange[0], zrange[1])) & (df_STDFIT.apply(pf.positive_def, axis=1))).values

df_BBC = df_BBC[mask]
df_STDFIT = df_STDFIT[mask]
df_STDFIT = df_STDFIT.apply(pf.x0_to_mB_err, axis=1)

############
# FIT DATA #
############

for su in np.arange(10, 25.5, 1.):
    timestarts = time.time()

    RES['NSN'].append(len(df_BBC))
    RES['sigu'].append(su)
    
    pw_dic_class = {'vv': [[kh, ptt * (np.sin(kh * su)/(kh * su))**2]]} 

    # TRUE FIT
    minuit_fitter = pf.fit_fs8_TRUE(
        df_BBC, 
        cosmo, 
        pw_dic_class,
        pf.parameter_dict_TRUE, 
        pf.likelihood_properties_BBC, 
        kmin,
    )

    RES = pf.fill_minuit_RES(RES, minuit_fitter, 'TRUE')

    # STANDARD FIT
    minuit_fitter = pf.fit_fs8_stdfit(
        df_STDFIT, 
        cosmo, 
        pw_dic_class, 
        pf.parameter_dict_STDFIT, 
        pf.likelihood_properties_STDFIT,
        kmin
        )

    RES = pf.fill_minuit_RES(RES, minuit_fitter, 'STDFIT')


    # BBC FIT
    minuit_fitter = pf.fit_fs8_fromBBC(
        df_BBC, 
        cosmo, 
        pw_dic_class,
        pf.parameter_dict_BBC, 
        pf.likelihood_properties_BBC, 
        kmin, 
        )

    RES = pf.fill_minuit_RES(RES, minuit_fitter, 'BBC')
    
    timeends = time.time()
    dtime =  timeends - timestarts
    print(f'fs8 fitted in {dtime//60:.0f}min{dtime%60:.0f}sec\n')
    print('###################################################\n\n')

pd.DataFrame(RES).to_csv(OUTPUT_DIR + f'RES_MOCK{mock:02d}_{model.upper()}_SIGU_zrange_{zrange[0]}_{zrange[1]}_nojax.csv')