#!/bin/bash
#SBATCH --job-name=SIMP_INT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=1g
#SBATCH --partition=all
#SBATCH --gres=gpu:1

python3 run.py
