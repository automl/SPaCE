#!/bin/bash
#
#SBATCH --partition=cpu_cluster
#SBATCH --job-name=vds_rr
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8000M

source ~/anaconda3/tmp/bin/activate vds

python -m baselines.run --alg=ppo2 --env=antgoal-v0 --num_timesteps=1000000 --log_path=../vds_results/vds_rr_new_s$1 --seed $1
