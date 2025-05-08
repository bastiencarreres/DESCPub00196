#!/bin/bash
#SBATCH -J FIT_FS8_COLOR
#SBATCH -A m1727
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --array=2-8
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=100GB
#SBATCH -e ./logs/REFORMAT_COV_%a.err
#SBATCH -o  ./logs/REFORMAT_COV_%a.out

source ~/.bashrc

conda activate flip_nojax

MOCK_NUMBER=$((--SLURM_ARRAY_TASK_ID))

python reformat_cov_mat.py $MOCK_NUMBER