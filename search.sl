#!/bin/bash 
#SBATCH --job-name=search_glitch
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --account=kriles0
#SBATCH --partition=standard
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/cwglitch/out/search4.out

source /home/damoncht/miniconda3/etc/profile.d/conda.sh

conda activate lalsuite-dev

python search.py --data_label wg_dnu_nu_1e-6_dnu1_nu1_1e-3_q_0.7 --result_label wg_dnu_nu_1e-6_dnu1_nu1_1e-3_q_0.7_f0f1 --df_grid 5e-4 5e-10 0 --cpus 16

echo "Search job finished"