"""This code generate the pippin config files for SNANA simulation and analyses."""

import numpy as np
import os

path = os.getenv('DESCPub00196')
MAIN_SEED = 151219970404202403081997
rand_gen = np.random.default_rng(MAIN_SEED)


YAML_TEXT = """# % include: $DESCPub00196/pippin_files/global_cfg/sim_aliases.yml %

ALIAS:
  MOCK_NUMBER: &mock_number 
    HOSTLIB_FILE: $SNANA_LSST_USERS/bastienc/UchuuCatalogs/UchuuDR2_UM/SNANA_SIMLIB/baseline_v3.3_10yrs/baseline_v3.3_10yrs_UchuuDR2_UM_z0p00_zmax0p1739_mock_{:02d}_SNANA.HOSTLIB
    SIMLIB_FILE: $SNANA_LSST_USERS/bastienc/UchuuCatalogs/UchuuDR2_UM/SNANA_SIMLIB/baseline_v3.3_10yrs/baseline_v3.3_10yrs_UchuuDR2_UM_z0p00_zmax0p1739_mock_{:02d}_SNANA.SIMLIB
  MOCK_RANDSEED: &mock_randseed
    RANSEED_REPEAT: 10 {}
SIM:
  LSST_P21:
    IA_SALT2_P21:
      <<: *salt2_p21
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_NO_DUST:
    IA_SALT2_P21_NO_DUST:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_NO_DUST.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_FIXED_BETA:
    IA_SALT2_P21_FIXED_BETA:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_FIXED_BETA.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_FIXED_BETA_NO_DUST:
    IA_SALT2_P21_FIXED_BETA_NO_DUST:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_FIXED_BETA_NO_DUST.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_REDUCED_BETA1:
    IA_SALT2_P21_REDUCED_BETA1:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_REDUCED_BETA_0_05.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_REDUCED_BETA2:
    IA_SALT2_REDUCED_BETA2:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_REDUCED_BETA_0_25.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_REDUCED_BETA3:
    IA_SALT2_REDUCED_BETA3:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_REDUCED_BETA_0_75.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_REDUCED_TAU1:
    IA_SALT2_P21_REDUCED_TAU1:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_REDUCED_TAU_0_05.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_REDUCED_TAU2:
    IA_SALT2_REDUCED_TAU2:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_REDUCED_TAU_0_25.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_REDUCED_TAU3:
    IA_SALT2_REDUCED_TAU3:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_REDUCED_TAU_0_75.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_REDUCED_TAU4:
    IA_SALT2_REDUCED_TAU4:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_REDUCED_TAU_0_50.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_REDUCED_TAU_BETA1:
    IA_SALT2_REDUCED_TAU_BETA1:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_REDUCED_TAU_BETA_0_05.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_REDUCED_TAU_BETA2:
    IA_SALT2_REDUCED_TAU_BETA2:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_REDUCED_TAU_BETA_0_25.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21_REDUCED_TAU_BETA3:
    IA_SALT2_REDUCED_TAU_BETA3:
      <<: *base_sim_p21
      GENPDF_FILE: $DESCPub00196/pippin_files/snana_input/P23_variations/DES-SN5YR_LOWZ_S3_P21_REDUCED_TAU_BETA_0_75.DAT
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed   
      
LCFIT:
  LSST_FIT:
    MASK: LSST
    # The base nml file used 
    BASE: $DESCPub00196/pippin_files/snana_input/fit_cfg.nml
    OPTS:
      BATCH_INFO: sbatch $SBATCH_TEMPLATES/SBATCH_DEFAULT.TEMPLATE 20
""".format

for mock_number in range(0, 8):
    with open(path + f'/pippin_files/mock_ymls/P21_variations/LSST_Uchuu_Mock{mock_number:02d}_P21_variations_BC.yml', 'w') as f:
        f.write(YAML_TEXT(mock_number, mock_number, rand_gen.integers(1000, 10_000)))