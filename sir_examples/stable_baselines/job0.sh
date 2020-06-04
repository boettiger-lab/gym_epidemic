#!/bin/bash
# Job name:
#SBATCH --job-name=auto_1e7
#
# Account:
#SBATCH --account=fc_cboettig
#
# Partition:
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
# Other info:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Wall clock limit:
#SBATCH --time=00:25:00
#
## Command(s) to run:
module load python/3.7
module load cuda/10.1
module load gcc/8.3.0
# curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# python get-pip.py
# rm get-pip.py
# pip install --upgrade pip
# pip install stable_baselines3
# pip install gym
source ~/.virtualenv/open_ai_covid19/bin/activate
python sb_sac_v3.py

