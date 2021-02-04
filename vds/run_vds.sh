#!/bin/bash
#
#SBATCH --partition=cpu_cluster
#SBATCH --job-name=vds
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8000M

source ~/anaconda3/tmp/bin/activate vds

python -m baselines.ve_run --alg=her --env=antgoal-v0 --num_timesteps=1000000 --size_ensemble=3 --log_path=../vds_results/vds_new_s$1 --seed $1
