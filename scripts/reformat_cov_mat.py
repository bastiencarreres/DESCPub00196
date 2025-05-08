import os, sys, gzip
import numpy  as np
from  pathlib import Path
from astropy.io import ascii

mock = int(sys.argv[1])

def build_covariance(filename):
    """Run once at the start to build the covariance matrix for the data"""
    print("Loading covariance from {}".format(filename))

    # The file format for the covariance has the first line as an integer
    # indicating the number of covariance elements, and the the subsequent
    # lines being the elements.
    # This function reads in the file and the nasty for loops trim down the covariance
    # to match the only rows of data that are used for cosmology

    f = gzip.open(filename)
    line = f.readline()
    n = int(line)
    C = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            C[i,j] = float(f.readline())
    f.close()

    print('Done')

    # Return the covariance; the parent class knows to invert this
    # later to get the precision matrix that we need for the likelihood.
    return C


# FILES
PIPPIN_DIR = Path(os.environ['PIPPIN_OUTPUT'])
MOCK_DIR = PIPPIN_DIR / f'LSST_UCHUU_MOCK{mock:02d}_BC'
INPUT_COV = MOCK_DIR / '7_CREATE_COV/UNBINNED_COV/output/UNBINNED_COV_LSST_P21_OUTPUT_BBCFIT/covsys_000.txt.gz'

data = ascii.read(MOCK_DIR / '7_CREATE_COV/UNBINNED_COV/output/UNBINNED_COV_LSST_P21_OUTPUT_BBCFIT/hubble_diagram.txt')

OUTPATH = '/global/homes/b/bastienc/MY_SNANA_DIR/LSST_SNANA/PaperBBCVpec/cov_and_data/'

# COMPUTE COV
COV = build_covariance(INPUT_COV)
COV = COV[np.triu_indices_from(COV)]
np.savez(OUTPATH + f'covsys_000_MOCK{mock:02d}', CID=data['CID'], COV=COV)