#!/bin/bash
#SBATCH -J FIT_FS8_ERRSCALE
#SBATCH -A m1727
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --array=1-8
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40GB
#SBATCH -e ./logs/FIT_FS8_ERRSCALE_NOJAX_%a.err
#SBATCH -o  ./logs/FIT_FS8_ERRSCALE_NOJAX_%a.out

source ~/.bashrc

conda activate flip_nojax

MOCK_NUMBER=$((--SLURM_ARRAY_TASK_ID))

python fit_fs8_err_scale.py $MOCK_NUMBER