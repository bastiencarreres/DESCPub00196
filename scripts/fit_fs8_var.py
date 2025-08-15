import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
import paper_fun as pf
import flip
from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)


pf.limit_numpy(4)

zrange = [0.02, 0.1]
mock = int(sys.argv[1])
OUTPUT_DIR = '/global/homes/b/bastienc/MY_SNANA_DIR/LSST_SNANA/DESCPub00196/results_p21_var/'

######################
# SET POWER SPECTRUM #
######################

kmin = (2 * np.pi) / 2000 # 2pi / L
kmax = 0.2

kh, ptt = pf.init_PS(1e-5, kmax)

su = 21 # fitted on true vel
pw_dic_class = {'vv': [[kh, ptt * (np.sin(kh * su)/(kh * su))**2]]} 


############
# SET PATH #
############

# VARIATIONS
KEYS = ['P21', 'P21_NODUST', 'P21_FIXEDBETA', 
        'P21_REDUCEDTAU1', 'P21_REDUCEDTAU2', 'P21_REDUCEDTAU3',
        'P21_REDUCEDBETA1', 'P21_REDUCEDBETA2', 'P21_REDUCEDBETA3',
        'P21_REDUCEDTAUBETA'
       ]

PIPPIN_DIR = Path(os.environ['PIPPIN_OUTPUT'])
MOCK_DIR = PIPPIN_DIR / 'LSST_UCHUU_MOCK00_P21_VARIATIONS_BC'

FIT_DIR =  MOCK_DIR / '2_LCFIT'

# ORIGINAL
MOCK_DIR_ORIGIN = PIPPIN_DIR / f'LSST_UCHUU_MOCK00_BC'
BBC_DIR_ORIGIN = MOCK_DIR_ORIGIN / '6_BIASCOR/LSST_P21/output'
BBC_FILE_ORIGIN = BBC_DIR_ORIGIN / f'OUTPUT_BBCFIT/FITOPT000_MUOPT000.FITRES.gz'


#####################
# COMPUTE USED CIDs #
#####################
# BBC DATA
df_BBC_ORIGIN = ascii.read(BBC_FILE_ORIGIN).to_pandas().set_index('CIDint')
BBC_mask = (df_BBC_ORIGIN["HOST_NMATCH"] > 0) & (df_BBC_ORIGIN['zHD'].between(zrange[0], zrange[1]))
df_BBC_ORIGIN = df_BBC_ORIGIN[BBC_mask]

CIDarr = df_BBC_ORIGIN.index.values
for k in KEYS:
    FIT_FILE = FIT_DIR /  f'LSST_FIT_LSST_{k}/output/PIP_LSST_UCHUU_MOCK00_P21_VARIATIONS_BC_LSST_{k}/FITOPT000.FITRES.gz'
    df_STDFIT = ascii.read(FIT_FILE).to_pandas().set_index('CID')
    df_STDFIT[df_STDFIT.apply(pf.positive_def, axis=1)]
    CIDarr = np.intersect1d(CIDarr, df_STDFIT.index)

print(f'PROP: {len(CIDarr) / len(df_BBC_ORIGIN.index.values):.3f}')

del df_BBC_ORIGIN
del df_STDFIT

###########
# RUN FIT #
###########

RES = pf.init_res_dic(fittypes=['TRUE', 'STDFIT'], add_keys=['model'])

for k in KEYS:
    print(f'STARTING MODEL {k}')
    RES['model'].append(k)
    
    timestarts = time.time()
    FIT_FILE = FIT_DIR /  f'LSST_FIT_LSST_{k}/output/PIP_LSST_UCHUU_MOCK00_P21_VARIATIONS_BC_LSST_{k}/FITOPT000.FITRES.gz'

    if not FIT_FILE.exists():
        raise ValueError(f'{FIT_FILE} does not exist')
        
    #############
    # LOAD DATA #
    #############

    # STANDARD DATA
    df_STDFIT = ascii.read(FIT_FILE).to_pandas().set_index('CID').loc[CIDarr]
    df_STDFIT = df_STDFIT.apply(pf.x0_to_mB_err, axis=1)
    
    RES['NSN'].append(len(df_STDFIT))
    
    
    ############
    # FIT DATA #
    ############
    
    # TRUE FIT
    minuit_fitter = pf.fit_fs8_TRUE(
        df_STDFIT, 
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
    
    timeends = time.time()
    dtime =  timeends - timestarts
    print(f'fs8 fitted in {dtime//60:.0f}min{dtime%60:.0f}sec\n')
    print('###################################################\n\n')

pd.DataFrame(RES).to_csv(OUTPUT_DIR + f'RES_MOCK00_P21_VAR_zrange_{zrange[0]}_{zrange[1]}_nojax.csv')