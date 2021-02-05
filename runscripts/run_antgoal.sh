#!/bin/bash
#
#SBATCH --partition=cpu_cluster
#SBATCH --job-name=ag_baselines
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8000M

python ../src/baselines_spl.py --env ant-goal --outdir antgoal_oj_spl$1 --algo ppo --mode spl --seed $1 --setfactor 1 --warmup 0 --test ../features/ag_oj_test.csv --instances ../features/ag_oj_train.csv
