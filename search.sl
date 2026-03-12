#!/bin/bash 
#SBATCH --job-name=sim_glitch
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --account=kriles1
#SBATCH --partition=standard
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/simGlitch/out/search.out

source /home/damoncht/miniconda3/etc/profile.d/conda.sh

conda activate lalsuite-dev

python search.py --label wglitch_f1_3e-9 --tcoh_list 5 10 20 40

echo "Search job finished"