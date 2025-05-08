"""Functions used in PV in BBC framework paper."""

import numpy as np
import iminuit
import astropy.constants as acst
import flip 
try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass
    
def limit_numpy(nthreads=4):
    """ """
    import os
    threads = str(nthreads)
    print(f"threads {threads}")
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads
    
#############
# CONSTANTS #
#############

__C_LIGHT_KMS__ = acst.c.to('km/s').value


# PARAMETERS FOR FLIP #

# -- COSMO for CLASS
cosmo_dic = {'h': 0.6774, 'sigma8': 0.8159, 'n_s': 0.9667}
cosmo_dic['omega_b'] = 0.0486 * cosmo_dic['h']**2
cosmo_dic['omega_cdm'] = (0.3089 - 0.0486) * cosmo_dic['h']**2


# -- BBC FIT
likelihood_properties_BBC = {
    "inversion_method": "cholesky",
    "use_jit": False,
    "use_gradient": False
}

parameter_dict_TRUE = {
    "fs8": {
        "value": 1.,
        "limit_low" : 0.0,
        "fixed" : False,
    },
    "sigv": {
        "value": 200,
        "limit_low" : 0.0,
        "fixed" : False,
    },
    
    "vmean" : {
        "value": 0,
        "fixed" : True,
    },
}

parameter_dict_BBC = {
    "fs8": {
        "value": 1.,
        "limit_low" : 0.0,
        "fixed" : False,
    },
    "M_0": {
         "value": 0.0,
        "fixed" : False,
        },
    "sigv": {
        "value": 200,
        "limit_low" : 0.0,
        "fixed" : False,
    },
    
    "vmean" : {
        "value": 0,
        "fixed" : True,
    },
}

# -- STD FIT

likelihood_properties_STDFIT = {
    "inversion_method": "cholesky",
    "use_jit": False,
    "use_gradient": False
}

parameter_dict_STDFIT = {
    "fs8": {
        "value": 1.,
        "limit_low" : 0.0,
        "fixed" : False,
        },                
    "sigv": {
        "value": 200,
        "limit_low" : 0.0,
        "fixed" : False,
        },
    "vmean": {
        "value": 0,
        "fixed" : True,
        },
    "alpha": {
        "value": 0.14,
        "fixed" : False,
        },
    "beta": {
        "value": 3.1,
        "fixed" : False,
        },
    "M_0": {
        "value": -19,
        "fixed" : False,
        },
    "sigma_M": {
        "value": 0.11,
        "fixed" : False,
        "limit_low" : 0.0,
        },
    "gamma": {
        "value": 0.025,
        "fixed" : False,
        },
    }

#######################
# Init Power Spectrum #
#######################

def init_PS(kmin, kmax):
    kh, _, _, ptt, _ = flip.power_spectra.compute_power_spectra(
    'class_engine',
    cosmo_dic, 
    0, kmin, 
    kmax, 
    1500, 
    normalization_power_spectrum="growth_rate"
    )
    return kh, ptt

##########
# Divers #
##########

def positive_def(x):
    cov = np.array([[x['x0ERR']**2, x['COV_x1_x0'], x['COV_c_x0']], 
                   [x['COV_x1_x0'], x['x1ERR']**2, x['COV_x1_c']], 
                   [x['COV_c_x0'], x['COV_x1_c'], x['cERR']**2]])
    return np.all(np.linalg.eigvals(cov) > 0)


def apply_cuts(df, z_range=[0.02, 0.1],
               x1_range=[-3, 3],
               c_range=[-0.3, 0.3],
               x1err_range=[0, 1],
               cerr_range=[0, 1.5],
               t0err_range=[0, 2.0],
               fitprob_range=[0.001, 1.1]):
    
    mask = np.ones(len(df), dtype='bool')
    if z_range is not None:
        mask &= df.zCMB.between(*z_range)
        print(f'Z_RANGE={z_range}')
    if x1_range is not None:
        mask &= df.x1.between(*x1_range)
        print(f'X1_RANGE={x1_range}')
    if c_range is not None:
        mask &= df.c.between(*c_range)
        print(f'C_RANGE={c_range}')
    if x1err_range is not None:
        mask &= df.x1ERR.between(*x1err_range)
        print(f'X1ERR_RANGE={x1err_range}')
    if cerr_range is not None:
        mask &= df.cERR.between(*cerr_range)
        print(f'CERR_RANGE={cerr_range}')
    if t0err_range is not None:
        mask &= df.PKMJDERR.between(*t0err_range)
        print(f'T0ERR_RANGE={t0err_range}')
    if fitprob_range is not None:
        mask &= df.FITPROB.between(*fitprob_range)
        print(f'FITPROB_RANGE={fitprob_range}')
    return mask
    

################
# x0 to mB err #
################
def x0_to_mB_err(x):
    cov = np.array([[x['x0ERR']**2, x['COV_x1_x0'], x['COV_c_x0']], 
                   [x['COV_x1_x0'], x['x1ERR']**2, x['COV_x1_c']], 
                  [x['COV_c_x0'], x['COV_x1_c'], x['cERR']**2]])
    J = np.array([[-2.5 / np.log(10) * 1 / x['x0'], 0, 0], [0, 1, 0], [0, 0, 1]])
    new_cov = J @ cov @ J.T
    x['COV_mB_mB'] = new_cov[0, 0]
    x['COV_mB_x1'] = new_cov[0, 1]
    x['COV_mB_c'] = new_cov[0, 2]
    return x

###################
# HD standard fit #
###################

def give_muth(z, cosmo, H0=70.):
    if cosmo == 'linear':
        print(f'Linear cosmo with H0 = {H0:.2f} km/s/Mpc')
        mu_th = 5 * np.log10(__C_LIGHT_KMS__ / H0 * (1 + z) * z) + 25
    else:
        mu_th = 5 * np.log10((1 + z) * cosmo.comoving_distance(z).value) + 25
    return mu_th

def compute_mu(mb, x1, c, e_mb, e_x1, e_c, 
               cov_mb_x1, cov_mb_c, cov_x1_c, 
               alpha, beta, M0, logmass=None, dM=None):
    
    mu = mb + alpha * x1 - beta * c - M0
    
    if logmass is not None and dM is not None:
        mask = logmass > 10
        mu[mask] -= dM / 2
        mu[~mask] += dM / 2
        
    mu_cov = e_mb**2 + alpha**2 * e_x1**2 + beta**2 * e_c**2
    mu_cov += 2 * alpha * cov_mb_x1 - 2 * beta * cov_mb_c
    mu_cov -= 2 * alpha * beta * cov_x1_c
    return mu, mu_cov


def fit_hd(z, mb, x1, c, e_mb, e_x1, e_c, 
           cov_mb_x1, cov_mb_c, cov_x1_c, 
           sigM, cosmo='linear', H0=70., logmass=None):
    
    mu_th = give_muth(z, cosmo, H0=H0)

    def chi2(alpha, beta, M0, dM=None):        
        mu, mu_cov = compute_mu(mb, x1, c, e_mb, e_x1, e_c, 
                                cov_mb_x1, cov_mb_c, cov_x1_c, 
                                alpha, beta, M0, logmass=logmass, dM=dM)
        mu_cov += sigM**2

        return ((mu - mu_th)**2 / mu_cov).sum()
    if logmass is not None:
        chimin = iminuit.Minuit(chi2, alpha=0.1, beta=3., M0=-19, dM=0.025)
        chimin.fixed = [False, False, False, False]
    else:
        chimin = iminuit.Minuit(chi2, alpha=0.1, beta=3., M0=-19, dM=0)
        chimin.fixed = [False, False, False, True]
        
    chimin.errordef = iminuit.Minuit.LEAST_SQUARES
    chimin.migrad()
    chimin.hesse()
    #chimin.minos()
    return chimin  

def fit_hd_REML(z, mb, x1, c, e_mb, e_x1, e_c,
                cov_mb_x1, cov_mb_c, cov_x1_c, 
                alpha, beta, M0, cosmo='linear', H0=70, logmass=None, dM=None):
    
    mu_th = give_muth(z, cosmo, H0=H0)

    def neg_like(sigM):
        mu, mu_cov = compute_mu(mb, x1, c, e_mb, e_x1, e_c, 
                                cov_mb_x1, cov_mb_c, cov_x1_c, 
                                alpha, beta, M0, logmass=logmass, dM=dM)    
        
        
        mu_cov += sigM**2
        chi2 = ((mu - mu_th)**2 / mu_cov).sum()
        loglik = np.sum(np.log(mu_cov)) + chi2 + np.log(np.sum(1/mu_cov))
        return loglik
    
    chimin = iminuit.Minuit(neg_like, sigM=0.15)
    chimin.errordef = iminuit.Minuit.LIKELIHOOD
    chimin.migrad()
    chimin.hesse()
    chimin.minos()
    return chimin 

# Iterative fit
def fit_HD_3steps(z, mb, x1, c, e_mb, e_x1, e_c,
                  cov_mb_x1, cov_mb_c, cov_x1_c,
                  cosmo='linear', nit=1, logmass=None, sigM_fix=None):
    
    

    if sigM_fix is None:
        sigM = 0.2
    else:
        sigM = sigM_fix
        
    chi = fit_hd(z, mb, x1, c, 
                 e_mb, e_x1, e_c, cov_mb_x1, cov_mb_c, cov_x1_c, sigM,
                 cosmo=cosmo, logmass=logmass)
    
    for i in range(nit):
        if sigM_fix is None:
            if logmass is None:
                lik_reml = fit_hd_REML(z, mb, x1, c, 
                                       e_mb, e_x1, e_c, cov_mb_x1, cov_mb_c, cov_x1_c,
                                       chi.values['alpha'], chi.values['beta'], chi.values['M0'], 
                                       cosmo=cosmo)
            else:
                lik_reml = fit_hd_REML(z, mb, x1, c, 
                                       e_mb, e_x1, e_c, cov_mb_x1, cov_mb_c, cov_x1_c,
                                       chi.values['alpha'], chi.values['beta'], chi.values['M0'], 
                                       logmass=logmass, dM=chi.values['dM'],
                                       cosmo=cosmo)
                
            sigM = lik_reml.values['sigM']
            
        chi = fit_hd(z, mb, x1, c, 
                     e_mb, e_x1, e_c, cov_mb_x1, cov_mb_c, cov_x1_c, sigM,
                     cosmo=cosmo, logmass=logmass)
        
    return lik_reml, chi

# Fitting with likelihood
def lik_fit_hd(z, mb, x1, c, e_mb, e_x1, e_c, cov_mb_x1, cov_mb_c, cov_x1_c,
               cosmo='linear', zerr=None, use_sim=False, H0=70., 
               init=[0.14, 3., -19., 0.1], logmass=None, dM=None):    
     
    mu_th = give_muth(z, cosmo, H0=H0)
    
    def neg_like(alpha, beta, M0, sigM, dM):
        mu, mu_cov = compute_mu(mb, x1, c, e_mb, e_x1, e_c, 
                                cov_mb_x1, cov_mb_c, cov_x1_c, 
                                alpha, beta, M0,
                                logmass=logmass, dM=dM)
        
        if zerr is not None:
            mu_cov += (5 / np.log(10) * zerr / z)**2 
            
        mu_cov += sigM**2
        
        chi2 = ((mu - mu_th)**2 / mu_cov).sum()
        loglik = 0.5 * (len(mu) * np.log(2 * np.pi) + np.sum(np.log(mu_cov)) + chi2)
        return loglik
    
    if logmass is not None:
        chimin = iminuit.Minuit(neg_like, alpha=init[0], beta=init[1], M0=init[2], sigM=init[3], dM=0.025)
        chimin.fixed = [False, False, False, False, False]
    else:
        chimin = iminuit.Minuit(neg_like, alpha=init[0], beta=init[1], M0=init[2], sigM=init[3], dM=0)
        chimin.fixed = [False, False, False, False, True]
        
    chimin.errordef = iminuit.Minuit.LIKELIHOOD
    chimin.migrad()
    chimin.hesse()
    chimin.minos()
    return chimin 

###########
# VEL EST #
###########

def VelEst(z, dmu, dmu_err, cosmo):
    rH = cosmo.H(z).value * cosmo.comoving_distance(z).value

    pfct = __C_LIGHT_KMS__ * np.log(10) / 5 
    zdep = ((1 + z) * __C_LIGHT_KMS__ / rH - 1.)**(-1)
    
    vest = -pfct * zdep * dmu
    verr = pfct * zdep * dmu_err
    return vest, verr

####################
# FIT FS8 FUNCTION #
####################

def init_res_dic(fittypes=['BBC', 'TRUE', 'STDFIT'], add_keys=None):
    RES = { 
        'NSN': [],
    }
    
    if add_keys is not None:
        for k in add_keys:
            RES[k] = []

    for t in fittypes:
        RES['valid_' + t] = []
        RES['accurate_' + t] = []

    if 'BBC' in fittypes:
        for k in parameter_dict_BBC:
            if not parameter_dict_BBC[k]['fixed']:
                RES[k + '_BBC'] = []
                RES[k + '_ERR_BBC'] = []
                RES[k + '_MERR_UP_BBC'] = []
                RES[k + '_MERR_LOW_BBC'] = []
                
    if 'BBCCOV' in fittypes:
        for k in parameter_dict_BBC:
            if not parameter_dict_BBC[k]['fixed']:
                RES[k + '_BBCCOV'] = []
                RES[k + '_ERR_BBCCOV'] = []
                RES[k + '_MERR_UP_BBCCOV'] = []
                RES[k + '_MERR_LOW_BBCCOV'] = []
    if 'TRUE' in fittypes:
        for k in parameter_dict_TRUE:
            if not parameter_dict_TRUE[k]['fixed']:
                RES[k + '_TRUE'] = []
                RES[k + '_ERR_TRUE'] = []
                RES[k + '_MERR_UP_TRUE'] = []
                RES[k + '_MERR_LOW_TRUE'] = []
    
    if 'STDFIT' in fittypes:
        for k in parameter_dict_STDFIT:
            if not parameter_dict_STDFIT[k]['fixed']:
                RES[k + '_STDFIT'] = [] 
                RES[k + '_ERR_STDFIT'] = [] 
                RES[k + '_MERR_UP_STDFIT'] = []
                RES[k + '_MERR_LOW_STDFIT'] = []

    return RES

def fill_minuit_RES(RES, minuit_fitter, fittype):
    RES['valid_' + fittype].append(minuit_fitter.minuit.valid)
    RES['accurate_' + fittype].append(minuit_fitter.minuit.accurate)

    if fittype in ['BBC', 'BBCCOV']:
        pdic = parameter_dict_BBC
    elif fittype == 'TRUE':
        pdic = parameter_dict_TRUE
    elif fittype == 'STDFIT':
        pdic = parameter_dict_STDFIT

    for p in pdic:
        if not pdic[p]['fixed']:
            RES[p + '_' + fittype].append(minuit_fitter.minuit.values[p])
            RES[p + '_ERR_' + fittype].append(minuit_fitter.minuit.errors[p])
            RES[p + '_MERR_UP_' + fittype].append(minuit_fitter.minuit.merrors[p].upper)
            RES[p + '_MERR_LOW_' + fittype].append(minuit_fitter.minuit.merrors[p].lower)
    return RES

    
def give_stdfit_data(df, cosmo):
    
    data_dic = {
        "ra": np.deg2rad(df['HOST_RA'].values),
        "dec": np.deg2rad(df['HOST_DEC'].values),
        "mb": df['mB'].values ,
        "x1": df["x1"].values,
        "c": df["c"].values,
        "e_mb": df["mBERR"].values,
        "e_x1": df["x1ERR"].values,
        "e_c": df["cERR"].values,
        "cov_mb_x1": df["COV_mB_x1"].values,
        "cov_mb_c": df["COV_mB_c"].values,
        "cov_x1_c": df["COV_x1_c"].values,
        "zobs": df["zHD"].values,
        "rcom_zobs": cosmo.comoving_distance(df["zCMB"].values).value * cosmo.h,
        "hubble_norm": 100 * cosmo.efunc(df["zCMB"].values),
        "host_logmass": df["HOST_LOGMASS"].values,
        }
    
    if len(df["HOST_OBJID"].unique()) != len(df["HOST_OBJID"]):
        data_dic["host_group_id"] = df["HOST_OBJID"].values
        
    return flip.data_vector.snia_vectors.VelFromSALTfit(data_dic, h=cosmo.h)

def give_bbc_data(df, cosmo, cov=None, err_scale=1.):
    dmu = df['MU'].values - cosmo.distmod(df['zHD'].values).value
    
    data_dic =  {
            "ra": np.deg2rad(df['HOST_RA'].values),
            "dec": np.deg2rad(df['HOST_DEC'].values),
            "dmu": dmu,
            "zobs": df['zHD'].values,
            "rcom_zobs": cosmo.comoving_distance(df["zHD"].values).value * cosmo.h,
            "hubble_norm": 100 * cosmo.efunc(df["zHD"].values),
        }
    
    if len(df["HOST_OBJID"].unique()) != len(df["HOST_OBJID"]):
        data_dic["host_group_id"] = df["HOST_OBJID"].values
    if cov is None:
        data_dic["dmu_error"] = df['MUERR'].values * np.sqrt(err_scale)
    else: 
        cov *= err_scale
    
    return flip.data_vector.basic.VelFromHDres(
        data_dic, 
        vel_estimator="full",
        covariance_observation=cov
        )

def fit_fs8_stdfit(df, cosmo, pw_dic_class, parameter_dict, likelihood_properties, kmin):
    print('Compute data vector')
    data = give_stdfit_data(df, cosmo)
    print('Compute COV')
    COV = data.compute_covariance(
        'carreres23',
        pw_dic_class,
        number_worker=20, 
        hankel=True, 
        kmin=kmin
    )
    print('Starts fit')
    minuit_fitter = flip.fitter.FitMinuit.init_from_covariance(
        COV, 
        data, 
        parameter_dict, 
        likelihood_properties=likelihood_properties
    )

    minuit_fitter.run(n_iter=3, hesse=True, minos=True)
    del COV
    return minuit_fitter


def fit_fs8_fromBBC(df, cosmo, pw_dic_class, parameter_dict, likelihood_properties, kmin, cov=None, nit=3, hesse=True, minos=True, err_scale=1.):
    print('Compute data vector')
    data = give_bbc_data(df, cosmo, cov=cov, err_scale=err_scale)    
    print('Compute COV')
    COV = data.compute_covariance(
        'carreres23',
        pw_dic_class,
        number_worker=20, 
        hankel=True, 
        kmin=kmin
    )
    print('Starts fit')
    minuit_fitter = flip.fitter.FitMinuit.init_from_covariance(
        COV, 
        data, 
        parameter_dict, 
        likelihood_properties=likelihood_properties)

    minuit_fitter.run(n_iter=nit, hesse=hesse, minos=minos)
    del COV
    return minuit_fitter



def fit_fs8_TRUE(df, cosmo, pw_dic_class, parameter_dict, likelihood_properties, kmin):
    print('Compute data vector')
    
    data_dic = {
            "ra": np.deg2rad(df['HOST_RA'].values),
            "dec": np.deg2rad(df['HOST_DEC'].values),
            "zobs": df['zHD'].values,
            "velocity":  df['SIM_VPEC'].values ,
            "velocity_error": np.ones(len(df)) * 1e-5,
            "rcom_zobs": cosmo.comoving_distance(df["zHD"].values).value * cosmo.h,
            "hubble_norm": 100 * cosmo.efunc(df["zHD"].values)
    }
    
    if len(df["HOST_OBJID"].unique()) != len(df["HOST_OBJID"]):
        data_dic["host_group_id"] = df["HOST_OBJID"].values
    
    data = flip.data_vector.basic.DirectVel(data_dic)
    
    print('Compute COV')
    COV = data.compute_covariance(
        'carreres23',
        pw_dic_class,
        number_worker=20, 
        hankel=True, 
        kmin=kmin
    )
    print('Starts fit')
    minuit_fitter = flip.fitter.FitMinuit.init_from_covariance(
        COV, 
        data, 
        parameter_dict, 
        likelihood_properties=likelihood_properties)

    minuit_fitter.run(n_iter=3, hesse=True, minos=True)
    del COV
    return minuit_fitter

    
