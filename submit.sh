#!/bin/bash
#SBATCH --job-name=REAL_TES
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=15g
#SBATCH --partition=all
#SBATCH --gres=gpu:1

export SUMO_HOME="/home/cgeortay/master_thesis/sumo/sumo"
python3 real_testing1.py
