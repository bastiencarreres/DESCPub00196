import os, sys, gzip
import numpy  as np
from  pathlib import Path
from astropy.io import ascii

mock = int(sys.argv[1])

# FILES
PIPPIN_DIR = Path(os.environ['PIPPIN_OUTPUT'])
MOCK_DIR = PIPPIN_DIR / f'LSST_UCHUU_MOCK{mock:02d}_BC'
BBC_DIR = MOCK_DIR / '6_BIASCOR/LSST_P21/output'
BBC_FILE = BBC_DIR / f'OUTPUT_BBCFIT/FITOPT000_{k[1]}.FITRES.gz'

IN_COV = f'/global/homes/b/bastienc/MY_SNANA_DIR/LSST_SNANA/PaperBBCVpec/cov_and_data/global/homes/b/bastienc/covsys_000_MOCK{mock:02d}.npz'

data = np.load(IN_COV)
df_cov_CID = data['CID']

df = ascii.read(BBC_FILE).to_pandas().set_index('CIDint')
df = df.loc[df_cov_CID]

cov = np.zeros((len(df), len(df)))
cov[np.triu_indices_from(cov)] = data['COV']
cov = cov.T + np.triu(cov, 1)



