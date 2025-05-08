#!/bin/bash
#SBATCH -J FIT_FS8_SIGU_P21
#SBATCH -A m1727
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --array=4-4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=80GB
#SBATCH -e ./logs/FIT_FS8_SIGU_NOJAX_%a.err
#SBATCH -o  ./logs/FIT_FS8_SIGU_NOJAX_%a.out

source ~/.bashrc

conda activate flip_nojax

MOCK_NUMBER=$((--SLURM_ARRAY_TASK_ID))

python fit_fs8_sigu_sys.py $MOCK_NUMBER 'P21'