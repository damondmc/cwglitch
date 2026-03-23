#!/bin/bash 
#SBATCH --job-name=sim_glitch
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --account=kriles0
#SBATCH --partition=standard
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/cwglitch/out/sim_real.out

source /home/damoncht/miniconda3/etc/profile.d/conda.sh

conda activate lalsuite-dev

python simulate.py --n_cpu 16 --label "wg_dnu_nu_1e-6_gaussian_gap" --freq 100.0 --sqrtSX 1e-22 --h0 1e-24

echo "Simulation job finished"
