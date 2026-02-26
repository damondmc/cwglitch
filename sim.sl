#!/bin/bash 
#SBATCH --job-name=sim_glitch
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=05:00:00
#SBATCH --account=kriles1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/%x-%j.log

source /home/damoncht/miniconda3/etc/profile.d/conda.sh

conda activate lalsuite-dev

python simulate.py --n_cpu 8 --label "with_glitch" --freq 100.0

echo "Simulation job finished"