import os
import glob
import subprocess
import numpy as np
from tqdm import tqdm

def f_lim(freq, tau=300., obsDay=240.):
    """Calculate the frequency limit for the narrowband search."""
    # limit for narrowband [f-lim/2,f+0.9+lim/2] needed for search using 1 year data, ref time in the middle of the search 
    # lim ~ fdot *(t-tref) and |fdot| <= f/tau, tau = 300,700, lim<= f*(t-tref)/tau = 0.5 * f/300 * Tobs/365
    # use 0.6 instead of 0.5 for antenna pattern buffer and multiple by the ration of total obs. day to a year
    lim = np.ceil(0.8 * (freq/tau) * (obsDay/365.) ) 
    return lim

def combine_sfts(fmin, fmax, fband, ts, te, output, sft_dir, fx=0.0):
    """
    Run lalpulsar_splitSFTs command for SFT files in a directory, sorted by timestamp.
    
    Parameters:
    - fmin (float): Minimum frequency
    - fmax (float): Maximum frequency
    - fband (float): Frequency band
    - fx (float): Frequency step
    - ts (int): Start time
    - te (int): End time
    - output (str): Output directory or filename prefix
    - sft_dir (str): Directory containing SFT files
    
    Returns:
    - None: Executes the command for each SFT file
    """
    # Get all .sft files in the directory
    sft_files = glob.glob(os.path.join(sft_dir, "*.sft"))
    
    # Sort files by timestamp (extracted from filename)
    # Assumes filename format like H-1_H1_1800SFT_simCW0-TIMESTAMP-DURATION.sft
    def get_timestamp(filename):
        # Extract the timestamp part (e.g., 1368970000 from H-1_H1_1800SFT_simCW0-1368970000-1800.sft)
        parts = os.path.basename(filename).split('-')
        if len(parts) >= 3:
            try:
                return int(parts[-2])  # Timestamp is second-to-last part
            except ValueError:
                return 0  # Fallback if timestamp is not an integer
        return 0
    
    sft_files = sorted(sft_files, key=get_timestamp)
    
    # Construct the command template
    cmd_template = (
        "lalpulsar_splitSFTs "
        "-fs {} -fe {} -fb {} -fx {} -ts {} -te {} -n {} -- {}"
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    
    # Run the command for each SFT file
    for sft in tqdm(sft_files, total=len(sft_files), desc="Combining SFTs..."):
        cmd = cmd_template.format(fmin, fmax, fband, fx, ts, te, output, sft)
        print(f"Executing: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for {sft}: {e}")
   
    # Run the command to remove sfts
    for sft in sft_files:
        try:
            os.remove(sft)  # Delete the SFT file after successful processing
        except OSError as e:
            print(f"Error removing file {sft}: {e}")