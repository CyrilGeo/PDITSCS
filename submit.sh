#!/bin/bash
#SBATCH --job-name=NETWORK
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=30g
#SBATCH --partition=all
#SBATCH --gres=gpu:1

export SUMO_HOME="/home/cgeortay/master_thesis/sumo/sumo"
python3 run_manhattan3.py
