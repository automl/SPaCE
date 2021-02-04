#!/bin/bash
#
#SBATCH --partition=cpu_cluster
#SBATCH --job-name=am_baselines
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8000M

source ~/anaconda3/tmp/bin/activate dac

python ../src/baselines_spl.py --env ant-maze --outdir antmaze$1_ppo64_rr$2 --algo ppo --mode rr --seed $2 --setfactor 1 --warmup 0 --test ../features/am_$1maze_test --instances ../features/am_$1maze_train
