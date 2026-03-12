#!/home/hoitim.cheung/.conda/envs/lalsuite-dev/bin/python
import os
import glob
import subprocess
from multiprocessing import Pool
import pandas as pd
import numpy as np
import argparse
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def search_range(m, tcoh, factor=1):
    """Calculate grid sizes for frequency and its derivatives."""
    df = 2 * np.sqrt(3*m) / np.pi / tcoh
    df1 = 12 * np.sqrt(5*m) / np.pi / tcoh**2
    df2 = 20 * np.sqrt(7*m) / np.pi / tcoh**3
    return [df*factor, df1*factor, df2*factor]

def find_sft_file(i, fmin, fmax, label, homedir):
    """Find the first .sft file in the data directory."""
    sft_pattern = os.path.join(homedir, f'data/{label}/{fmin}-{fmax}Hz/simCW{i}/*.sft')
    sft_files = glob.glob(sft_pattern)
    if not sft_files:
        raise FileNotFoundError(f"No .sft files found for simCW{i}")
    return sft_files[0]

def run_command(args):
    """Run a single lalpulsar_Weave command."""
    # Unpack all variables passed from main
    i, out_dir, metric_file, label, n_glitch, ip, df_grid, fmin, fmax, homedir, config = args
    
    sft_file = find_sft_file(i, fmin, fmax, label, homedir)
    
    idx = i * n_glitch
    
    # Logic: if grid value != 0, use range search; else use point search (/0)
    def get_range_str(val, df):
        return f"{val - df}/{2*df}" if df != 0 else f"{val}/0"

    freq_arg = get_range_str(ip['f0'][idx], df_grid[0])
    f1_arg   = get_range_str(ip['f1'][idx], df_grid[1])
    f2_arg   = get_range_str(ip['f2'][idx], df_grid[2])

    command = (
        f"lalpulsar_Weave "
        f"--output-file={out_dir}/CW{i}.fts "
        f"--sft-files='{sft_file}' "
        f"--setup-file={metric_file} "
        f"--semi-max-mismatch={config['semi_mm']} "
        f"--coh-max-mismatch={config['coh_mm']} "
        f"--toplist-limit={config['num_toplist']} "
        f"--extra-statistics='coh2F_det,mean2F,coh2F_det,mean2F_det' "
        f"--alpha={ip['alpha'][idx]}/0 "
        f"--delta={ip['delta'][idx]}/0 "
        f"--freq={freq_arg} "
        f"--f1dot={f1_arg} "
        f"--f2dot={f2_arg}"
    )

    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

    return result.stdout

def main():
    parser = argparse.ArgumentParser(description="Run lalpulsar_Weave commands in parallel.")
    parser.add_argument('--data_label', default='ng', help="Label for data directory")
    parser.add_argument('--result_label', default='ng', help="Label for data directory")
    parser.add_argument('--cpus', type=int, default=16, help="CPUs for multiprocessing")
    parser.add_argument('--fmin', type=int, default=100)
    parser.add_argument('--fmax', type=int, default=100)
    parser.add_argument('--n_glitch', type=int, default=1)
    parser.add_argument('--tcoh_list', type=int, nargs='+', default=[5, 10, 20, 40], help="List of coherence times in days")
    parser.add_argument('--df_grid', type=float, nargs=3, help="[df, df1, df2]")
    parser.add_argument('--homedir', default='/scratch/kriles_root/kriles0/damoncht/simGlitch')
    args = parser.parse_args()
    
    with open(f'{args.homedir}/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    csv_path = f'{args.homedir}/data/{args.data_label}/{args.fmin}-{args.fmax}Hz/signal_glitch_params.csv'
    ip = pd.read_csv(csv_path)
    
    n_signals = int(ip['n_th_signal'].max()) + 1
    logger.info(f"Detected {n_signals} signals from CSV.")
    
    params = ['f0', 'f1', 'f2']
    
    for tday in args.tcoh_list:
        logger.info(f"--- Starting Search for Tcoh = {tday} days ---")
        
        tcoh = 86400 * tday
        df_grid = args.df_grid if args.df_grid else search_range(config['coh_mm'], tcoh, factor=10)
        
        for p, val in zip(params, df_grid):
            mode = "RANGE" if val != 0 else "POINT"
            logger.info(f"{p:<5} : {val:>10.2e} ({mode})")
            
        # Setup directories
        out_dir = os.path.join(args.homedir, 'results', f'{tday}d', args.result_label, f'{args.fmin}-{args.fmax}Hz')
        os.makedirs(out_dir, exist_ok=True)
        
        metric_file = f'{args.homedir}/metric/metric_{tday}d.fts'
        
        # Validate metric file exists before launching
        if not os.path.exists(metric_file):
            logger.warning(f"Metric file {metric_file} missing! Skipping {tday}d.")
            continue

        # Prepare arguments
        command_args = [
            (i, out_dir, metric_file, args.data_label, args.n_glitch, ip, df_grid, args.fmin, args.fmax, args.homedir, config) 
            for i in range(n_signals)
        ]

        # Execute parallel pool for THIS tcoh
        logger.info(f"Launching pool for {tday}d using {args.cpus} cores...")
        with Pool(processes=args.cpus) as pool:
            pool.map(run_command, command_args)
            
        logger.info(f"Finished Tcoh = {tday} days.")

    print("All requested Tcoh searches completed.")


if __name__ == "__main__":
    main()