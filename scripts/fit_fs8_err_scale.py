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

zrange = [0.02, 0.1]
mock = int(sys.argv[1])
print(f"PROCESSIGN MOCK {mock:02d}\n\n")
OUTPUT_DIR = '/global/homes/b/bastienc/MY_SNANA_DIR/LSST_SNANA/PaperBBCVpec/results_err_scale/'

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

KEYS = ['C11', 'G10', 'P21', 'RNDSMEAR']

PIPPIN_DIR = Path(os.environ['PIPPIN_OUTPUT'])
MOCK_DIR = PIPPIN_DIR / f'LSST_UCHUU_MOCK{mock:02d}_BC'

BBC_DIR = MOCK_DIR / '6_BIASCOR'
FIT_DIR =  MOCK_DIR / '2_LCFIT'


###########
# RUN FIT #
###########

RES = pf.init_res_dic(fittypes=['BBC'], add_keys=['model', 'err_scale'])
for k in KEYS:
    print(f'STARTING MODEL {k}')
    timestarts = time.time()
    BBC_FILE = BBC_DIR / f'LSST_{k}/output/OUTPUT_BBCFIT/FITOPT000_MUOPT000.FITRES.gz'
    FIT_FILE = FIT_DIR /  f'LSST_FIT_LSST_{k}/output/PIP_LSST_UCHUU_MOCK{mock:02d}_BC_LSST_{k}/FITOPT000.FITRES.gz'
    
    if not BBC_FILE.exists():
        raise ValueError(f'{BBC_FILE} does not exist')
    if not FIT_FILE.exists():
        raise ValueError(f'{FIT_FILE} does not exist')
        
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
    
    data = pf.give_bbc_data(df_BBC, cosmo, cov=None, err_scale=1.)
    
    COV = data.compute_covariance(
        'carreres23',
        pw_dic_class,
        number_worker=20, 
        hankel=True, 
        kmin=kmin
    )
    
    for scale in np.arange(0.1, 2.0, 0.1):  

        RES['model'].append(k)
        RES['NSN'].append(len(df_BBC))
        RES['err_scale'].append(scale)
        
        ############
        # FIT DATA #
        ############
        data = pf.give_bbc_data(df_BBC, cosmo, cov=None, err_scale=scale)

        # BBC FIT
        minuit_fitter = flip.fitter.FitMinuit.init_from_covariance(
            COV, 
            data, 
            pf.parameter_dict_BBC, 
            likelihood_properties= pf.likelihood_properties_BBC)

        minuit_fitter.run(n_iter=3, hesse=True, minos=True)
        
        RES = pf.fill_minuit_RES(RES, minuit_fitter, 'BBC')

    timeends = time.time()
    dtime =  timeends - timestarts
    print(f'fs8 fitted in {dtime//60:.0f}min{dtime%60:.0f}sec\n')
    print('###################################################\n\n')

pd.DataFrame(RES).to_csv(OUTPUT_DIR + f'RES_MOCK{mock:02d}_zrange_{zrange[0]}_{zrange[1]}_nojax.csv')