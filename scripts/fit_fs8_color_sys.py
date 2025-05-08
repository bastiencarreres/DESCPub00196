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
print(f"PROCESSIGN MOCK {mock:02d}\n\n")
OUTPUT_DIR = '/global/homes/b/bastienc/MY_SNANA_DIR/LSST_SNANA/PaperBBCVpec/results_p21_ccut/'
MUOPT = ['INTRSC_BS20', 'INTRSC_P21SYS1', 'INTRSC_P21SYS2', 'INTRSC_P21SYS3']

######################
# SET POWER SPECTRUM #
######################

kmin = (2 * np.pi) / 2000 # 2pi / L
kmax = 0.2

kh, ptt = pf.init_PS(1e-5, kmax)

su = 21
pw_dic_class = {'vv': [[kh, ptt * (np.sin(kh * su)/(kh * su))**2]]} 


############
# SET PATH #
############

PIPPIN_DIR = Path(os.environ['PIPPIN_OUTPUT'])
MOCK_DIR = PIPPIN_DIR / f'LSST_UCHUU_MOCK{mock:02d}_BC'
BBC_DIR = MOCK_DIR / '6_BIASCOR/LSST_P21/output'
FIT_DIR =  MOCK_DIR / '2_LCFIT'
COV_DIR = Path('/global/homes/b/bastienc/MY_SNANA_DIR/LSST_SNANA/PaperBBCVpec/cov_and_data')


with open(BBC_DIR / 'SUBMIT.INFO', 'r') as stream:
    data_loaded = yaml.safe_load(stream)
    
muopt_to_process = [('DEFAULT_NOCOV', 'MUOPT000'), ('DEFAULT', 'MUOPT000')]

    
############
# INIT FIT #
############

RES = pf.init_res_dic(fittypes=['BBC', 'BBCCOV', 'TRUE', 'STDFIT'], add_keys=['ebv_cut', 'ebv_side'])

COV_FILE = COV_DIR / f'covsys_000_MOCK{mock:02d}.npz'
data = np.load(COV_FILE)
IDX_TO_USE = data['data']['CID']

BBC_FILE = BBC_DIR / f'OUTPUT_BBCFIT/FITOPT000_MUOPT000.FITRES.gz'
FIT_FILE = FIT_DIR /  f'LSST_FIT_LSST_P21/output/PIP_LSST_UCHUU_MOCK{mock:02d}_BC_LSST_P21/FITOPT000.FITRES.gz'


#############
# LOAD DATA #
#############

# BBC DATA
df_BBC = ascii.read(BBC_FILE).to_pandas().set_index('CIDint')
df_BBC = df_BBC.loc[IDX_TO_USE]

# STANDARD DATA
df_STDFIT = ascii.read(FIT_FILE).to_pandas().set_index('CID')
df_STDFIT = df_STDFIT.loc[df_BBC.index]

# MASK
mask = ((df_BBC["HOST_NMATCH"] > 0) & (df_BBC['zHD'].between(zrange[0], zrange[1])) & (df_STDFIT.apply(pf.positive_def, axis=1))).values

df_BBC = df_BBC[mask]
df_STDFIT = df_STDFIT[mask]
df_STDFIT = df_STDFIT.apply(pf.x0_to_mB_err, axis=1)

# COV
cov = np.zeros((len(mask), len(mask)))
cov[np.triu_indices_from(cov)] = data['cov']
cov = cov.T + np.triu(cov, 1)
cov = cov[np.ix_(mask, mask)]


############
# FIT DATA #
############

for ebv_cut in np.round(np.geomspace(0.01, 0.1, 11), 3):
    timestarts = time.time()
    cmask = df_BBC["SIM_AV"] / df_BBC["SIM_RV"] >= ebv_cut
    
    for CTYPE, MASK in zip(['high_ebv', 'low_ebv'], [cmask, ~cmask]): 
        N = np.sum(MASK)
        RES['NSN'].append(N)
        RES['ebv_cut'].append(ebv_cut)
        RES['ebv_side'].append(CTYPE)
        
        df_BBC_c = df_BBC[MASK]
        df_STDFIT_c =  df_STDFIT[MASK]
        cov_c = cov[np.ix_(MASK, MASK)]
        
        # TRUE FIT
        minuit_fitter = pf.fit_fs8_TRUE(
            df_BBC_c, 
            cosmo, 
            pw_dic_class,
            pf.parameter_dict_BBC, 
            pf.likelihood_properties_BBC, 
            kmin,
        )
    
        RES = pf.fill_minuit_RES(RES, minuit_fitter, 'TRUE')
    
        # STANDARD FIT
        minuit_fitter = pf.fit_fs8_stdfit(
            df_STDFIT_c, 
            cosmo, 
            pw_dic_class, 
            pf.parameter_dict_STDFIT, 
            pf.likelihood_properties_STDFIT,
            kmin
            )
    
        RES = pf.fill_minuit_RES(RES, minuit_fitter, 'STDFIT')
    
    
        # BBC FIT
        minuit_fitter = pf.fit_fs8_fromBBC(
            df_BBC_c, 
            cosmo, 
            pw_dic_class,
            pf.parameter_dict_BBC, 
            pf.likelihood_properties_BBC, 
            kmin, 
            )
    
        RES = pf.fill_minuit_RES(RES, minuit_fitter, 'BBC')
    
        # BBCCOV FIT
        minuit_fitter = pf.fit_fs8_fromBBC(
            df_BBC_c, 
            cosmo, 
            pw_dic_class,
            pf.parameter_dict_BBC, 
            pf.likelihood_properties_BBC, 
            kmin, 
            cov=cov_c
            )
    
        RES = pf.fill_minuit_RES(RES, minuit_fitter, 'BBCCOV')
    
    timeends = time.time()
    dtime =  timeends - timestarts
    print(f'fs8 fitted in {dtime//60:.0f}min{dtime%60:.0f}sec\n')
    print('###################################################\n\n')

pd.DataFrame(RES).to_csv(OUTPUT_DIR + f'RES_MOCK{mock:02d}_P21_C_CUT_zrange_{zrange[0]}_{zrange[1]}_nojax.csv')