"""This code generate the pippin config files for SNANA simulation and analyses."""

import numpy as np

MAIN_SEED = 151219970404202403081997
rand_gen = np.random.default_rng(MAIN_SEED)


YAML_TEXT = """# % include: $DESCPub00196/pippin_files/global_cfg/sim_aliases.yml %
# % include: $DESCPub00196/pippin_files/global_cfg/analysis_pipeline.yml %
# % include: $DESCPub00196/pippin_files/global_cfg/bbc_aliases.yml %
# % include: $DESCPub00196/pippin_files/global_cfg/external_aliases.yml %


ALIAS:
  MOCK_NUMBER: &mock_number 
    HOSTLIB_FILE: $SNANA_LSST_USERS/bastienc/UchuuCatalogs/UchuuDR2_UM/SNANA_SIMLIB/baseline_v3.3_10yrs/baseline_v3.3_10yrs_UchuuDR2_UM_z0p00_zmax0p1739_mock_{:02d}_SNANA.HOSTLIB
    SIMLIB_FILE: $SNANA_LSST_USERS/bastienc/UchuuCatalogs/UchuuDR2_UM/SNANA_SIMLIB/baseline_v3.3_10yrs/baseline_v3.3_10yrs_UchuuDR2_UM_z0p00_zmax0p1739_mock_{:02d}_SNANA.SIMLIB
  MOCK_RANDSEED: &mock_randseed
    RANSEED_REPEAT: 10 {}
SIM:
  LSST_RNDSMEAR:
    IA_SALT2_RNDSMEAR:
      <<: *salt2_rndsmear
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_G10:
    IA_SALT2_G10:
      <<: *salt2_g10
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_C11:
    IA_SALT2_C11:
      <<: *salt2_c11
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  LSST_P21:
    IA_SALT2_P21:
      <<: *salt2_p21
      <<: *mock_number
    GLOBAL:
      <<: *global_simpar
      <<: *mock_randseed
  <<: *external_bbc_sim
""".format

for mock_number in range(0, 8):
    with open(f'./mock_ymls/LSST_Uchuu_Mock{mock_number:02d}_BC.yml', 'w') as f:
        f.write(YAML_TEXT(mock_number, mock_number, rand_gen.integers(1000, 10_000)))