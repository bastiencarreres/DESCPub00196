from pathlib import Path

CONFIG_FILE_TEXT ="""INPUT_DIR:  /pscratch/sd/d/desctd/PIPPIN_OUTPUT/LSST_UCHUU_MOCK{0:02d}_BC/6_BIASCOR/LSST_P21/output
VERSION: OUTPUT_BBCFIT
OUTDIR: /pscratch/sd/d/desctd/PIPPIN_OUTPUT/LSST_UCHUU_MOCK{0:02d}_BC/7_CREATE_COV/UNBINNED_COV/output/UNBINNED_COV_LSST_P21_OUTPUT_BBCFIT
COSMOMC_DATASET_FILE: dataset.txt
COVOPTS:
- '[NOSYS]  [=DEFAULT,=DEFAULT]'
- '[BS20] [=DEFAULT,=INTRSC_BS20,1.0]'
- '[P21SYS1] [=DEFAULT,=INTRSC_P21SYS1,0.577]'
- '[P21SYS2] [=DEFAULT,=INTRSC_P21SYS2,0.577]'
- '[P21SYS3] [=DEFAULT,=INTRSC_P21SYS3,0.577]'
""".format

SBATCH_FILE_TEXT="""#!/bin/bash
#SBATCH -J LSST_UCHUU_MOCK{0:02d}_BC_CREATE_COV_UNBINNED_COV
#SBATCH -A m1727
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --output=UNBINNED_COV.OUT
#SBATCH --error=UNBINNED_COV.LOG

source /global/cfs/cdirs/lsst/groups/TD/setup_td.sh
cd /pscratch/sd/d/desctd/PIPPIN_OUTPUT/LSST_UCHUU_MOCK{0:02d}_BC/7_CREATE_COV/UNBINNED_COV/output/SCRIPTS_COVMAT

sh JOB.CMD
""".format

CMD_FILE_TEXT = """cd /pscratch/sd/d/desctd/PIPPIN_OUTPUT/LSST_UCHUU_MOCK{0:02d}_BC/7_CREATE_COV/UNBINNED_COV/output/SCRIPTS_COVMAT 

python /global/homes/b/bastienc/MY_SNANA_DIR/LSST_SNANA/PaperBBCVpec/pippin_files/create_covariance.py /pscratch/sd/d/desctd/PIPPIN_OUTPUT/LSST_UCHUU_MOCK{0:02d}_BC/7_CREATE_COV/UNBINNED_COV/input_file.txt \
   --yaml_file COVMAT_UNBINNED_COV_LSST_P21_OUTPUT_BBCFIT.YAML \
   --unbinned \
   --write_mask_cov 1 \
   --write_format npz \
  &>  COVMAT_UNBINNED_COV_LSST_P21_OUTPUT_BBCFIT.LOG 
  
touch COVMAT_UNBINNED_COV_LSST_P21_OUTPUT_BBCFIT.DONE 
echo 'Finished create_covariance.py -> create COVMAT_UNBINNED_COV_LSST_P21_OUTPUT_BBCFIT.DONE'
""".format

for mock in range(0,8):
    OUTPATH = Path(f"/pscratch/sd/d/desctd/PIPPIN_OUTPUT/LSST_UCHUU_MOCK{mock:02d}_BC/7_CREATE_COV/UNBINNED_COV")
    
    OUTPATH_COV = OUTPATH / 'output/UNBINNED_COV_LSST_P21_OUTPUT_BBCFIT'
    OUTPATH_COV.mkdir(parents=True, exist_ok=True)
    OUTPATH_SCRIPTS = OUTPATH / 'output/SCRIPTS_COVMAT' 
    OUTPATH_SCRIPTS.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPATH_SCRIPTS / 'JOB.CMD', 'w') as f:
        f.write(CMD_FILE_TEXT(mock))
    
    with open(OUTPATH_SCRIPTS / 'JOB.BATCH', 'w') as f:
        f.write(SBATCH_FILE_TEXT(mock))
        
    with open(OUTPATH / 'input_file.txt', 'w') as f:
        f.write(CONFIG_FILE_TEXT(mock))

