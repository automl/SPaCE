#!/bin/bash
#
#SBATCH --partition=gpu_cluster_enife
#SBATCH --job-name=cpm_spdl
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10000M
#SBATCH --gres=gpu:1

source ~/anaconda3/tmp/bin/activate dac

python ../spdl/run_experiment.py --seed $2 --learner trpo --type self_paced --env point_mass --train ../features/cpm_test_specific_$1.csv --test ../features/cpm_train_specific_$1.csv --base_log_dir spdl_trpo

