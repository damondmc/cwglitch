#!/home/hoitim.cheung/.conda/envs/lalsuite-dev/bin/python
import os
import glob
import subprocess
from multiprocessing import Pool
import pandas as pd
import numpy as np
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def grid_size(m, tcoh, factor=1):
    """Calculate grid sizes for frequency and its derivatives.

    Args:
        m (float): Some parameter related to the grid calculation.
        T (float): Observation time.
        factor (float, optional): Scaling factor for the grid sizes. Defaults to 1.

    Returns:
        list: Grid sizes for frequency, first derivative, and second derivative.
    """
    df = 2 * np.sqrt(3*m) / np.pi / tcoh
    df1 = 12 * np.sqrt(5*m) / np.pi / tcoh**2
    df2 = 20 * np.sqrt(7*m) / np.pi / tcoh**3
    df, df1, df2 = 2e-5, 1e-10, 5e-19
    #return [df*factor, df1*factor, df2*factor]
    return [df, df1, df2]

def find_sft_file(i, fmin, fmax, label, homedir):
    """Find the first .sft file in the data directory for given index and label.

    Args:
        i (int): Index of the simulation.
        fmin (int): Minimum frequency.
        fmax (int): Maximum frequency.
        label (str): Label for the data directory.
        homedir (str): Home directory path.

    Returns:
        str: Path to the first .sft file found.

    Raises:
        FileNotFoundError: If no .sft files are found.
    """
    sft_pattern = os.path.join(homedir, f'data/{label}/{fmin}-{fmax}Hz/simCW{i}/*.sft')
    sft_files = glob.glob(sft_pattern)
    if not sft_files:
        raise FileNotFoundError(f"No .sft files found in {os.path.join(homedir, f'data/{label}/simCW{i}/')}")
    return sft_files[0]

def run_command(args):
    """Run a single lalpulsar_Weave command."""
    i, out_dir, label, n_glitch, df, dx, tcoh_day = args
    try:
        sft_file = find_sft_file(i, fmin, fmax, label, homedir)
        command = (
            f"lalpulsar_Weave "
            f"--output-file={out_dir}/{label}_CW{i}.fts "
            f"--sft-files={sft_file} "
            f"--setup-file={metric_file} "
            f"--semi-max-mismatch={semimm} "
            f"--coh-max-mismatch={cohmm} "
            f"--toplist-limit={numtoplist} "
            f"--extra-statistics='coh2F_det,mean2F,coh2F_det,mean2F_det' "
            f"--alpha={df['alpha'][i*n_glitch]}/0 "
            f"--delta={df['delta'][i*n_glitch]}/0 "
            f"--freq={df['f0'][i*n_glitch]-dx[0]}/{2*dx[0]} "
            f"--f1dot={df['f1'][i*n_glitch]-dx[1]}/{2*dx[1]} "
            f"--f2dot={df['f2'][i*n_glitch]-dx[2]}/{2*dx[2]}"
#            f"--freq={df['f0'][i*n_glitch]}/0 "
#            f"--f1dot={df['f1'][i*n_glitch]}/0 "
#            f"--f2dot={df['f2'][i*n_glitch]}/0"
        )
        print(command)
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Command {i} for {label} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command {i} for {label} failed: {e.stderr}")
        return None
    except FileNotFoundError as e:
        logger.error(f"Command {i} for {label} failed: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run lalpulsar_Weave commands with multiprocessing.")
    parser.add_argument('--label', default='ng', help="Label for data directory")
    parser.add_argument('--cpus', type=int, default=16, help="Number of CPU cores to use for multiprocessing")
    parser.add_argument('--fmin', type=int, default=100, help="Min. frequency.")
    parser.add_argument('--fmax', type=int, default=100, help="Max. frequency.")
    parser.add_argument('--n_glitch', type=int, default=1, help="Number of glitches per signal.")
    parser.add_argument('--tcoh_day', type=int, default=5, help="Coherence time in day.")
    parser.add_argument('--homedir', default='/scratch/kriles_root/kriles0/damoncht/simGlitch', help="Base directory path")
    parser.add_argument('--n', type=int, default=16, help="Number of jobs (default: 16)")
    args = parser.parse_args()
    
    n_glitch = args.n_glitch
    tcoh = 86400 * args.tcoh_day
    dx = grid_size(m=0.1, tcoh=tcoh, factor=8)

    csv_path = f'{args.homedir}/data/{args.label}/{fmin}-{fmax}Hz/signal_glitch_params.csv'
    df = pd.read_csv(csv_path)

    # Create output directory
    out_dir = os.path.join(homedir, 'results', f'{args.tcoh_day}d', args.label, f'{fmin}-{fmax}Hz')
    os.makedirs(out_dir, exist_ok=True)

    # Prepare arguments for multiprocessing
    command_args = [(i, out_dir, args.label, n_glitch, df, dx, tcoh_day) for i in range(args.n)]

    # Run commands in parallel
    with Pool(processes=args.cpus) as pool:
        pool.map(run_command, command_args)
        
    print("All commands completed")


if __name__ == "__main__":
    main()
    