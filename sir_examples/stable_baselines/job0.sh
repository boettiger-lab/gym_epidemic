#!/bin/bash
# Job name:
#SBATCH --job-name=auto_1e7
#
# Account:
#SBATCH --account=fc_cboettig
#
# Partition:
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal
# Other info:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Wall clock limit:
#SBATCH --time=01:00:00
#
## Command(s) to run:
module load python/3.7
source ~/.virtualenv/open_ai_covid19/bin/activate
python sb_sac_v3.py
