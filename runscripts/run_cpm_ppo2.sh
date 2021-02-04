#!/bin/bash
#
#SBATCH --partition=cpu_cluster
#SBATCH --job-name=pm_baselines
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=1000M

source ~/anaconda3/tmp/bin/activate dac

python ../src/baselines_spl.py --outdir cpm_trpo$1_nonnorm_cl$2 --kappa 1 --eta 0.1 --mode cl --seed $2 --setfactor 1 --warmup 0 --env pointmass-gate --test ../features/cpm_test_$1.csv --instances ../features/cpm_train_$1.csv
