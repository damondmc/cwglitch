#!/bin/bash 
#SBATCH --job-name=sim_glitch
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --account=kriles1
#SBATCH --partition=standard
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/simGlitch/out/sim_1e-8.out

source /home/damoncht/miniconda3/etc/profile.d/conda.sh

conda activate lalsuite-dev

python simulate.py --n_cpu 16 --label "wg_dnu_nu_1e-5_q0.3" --freq 100.0

echo "Simulation job finished"
