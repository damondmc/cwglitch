#!/home/hoitim.cheung/.conda/envs/lalsuite-dev/bin/python
import os
import shutil
import lal
from lalpulsar import simulateCW
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from params import calc_absolute_glitch_params
from sft import combine_sfts


def waveform(h0, cosi, freq, f1dot, f2dot, f3dot, f4dot, glitch_params_norm):
    """Generate GW waveform for a pulsar with glitches."""
    for gp in glitch_params_norm:
        if len(gp) != 5:
            raise ValueError("Each absolute glitch parameter set must contain 5 elements: "
                             "[tglitch, df_p, df_t, df1_p, tau]")
    if f1dot == 0:
        raise ValueError("F1dot is zero.")

    f0 = freq
    f1dot0 = f1dot
    h0_scale = h0 / np.sqrt(np.abs(f1dot0) / f0)
    
    def wf(dt):
        dphi = freq * dt + f1dot * (1./2.) * dt**2 + f2dot * (1./6.) * dt**3 + \
               f3dot * (1./24.) * dt**4 + f4dot * (1./120.) * dt**5
        
        f_eff = f0
        f1dot_eff = f1dot0
        
        for gp in glitch_params_norm:
            tglitch, df_p, df_t, df1_p, tau = gp
            if dt > tglitch:
                delta_t = dt - tglitch
                dphi += df_p * delta_t 
                dphi += df_t * tau * (1 - np.exp(-delta_t / tau)) 
                dphi += df1_p * 0.5 * delta_t**2
                
                f_eff += df_p + df_t * np.exp(-delta_t / tau)
                f1dot_eff += df1_p - df_t / tau * np.exp(-delta_t / tau)
        
        if len(glitch_params_norm):
            if f1dot_eff == 0:
                raise ValueError("Effective f1dot is zero.")
            h0_t = h0_scale * np.sqrt(np.abs(f1dot_eff) / f_eff) 
        else:
            h0_t = h0
        
        dphi = lal.TWOPI * dphi
        ap = h0_t * (1.0 + cosi**2) / 2.0
        ax = h0_t * cosi
        return dphi, ap, ax
    
    return wf

def simulate_signal(signal_params):
    """Simulate a single continuous wave signal and generate SFT files."""
    tref = signal_params['tref']
    freq_params = signal_params['freq_params']
    phi0 = signal_params['phi0']
    psi = signal_params['psi']
    cosi = signal_params['cosi']
    alpha = signal_params['alpha']
    delta = signal_params['delta']
    glitch_params = signal_params['glitch_params']
    
    # Observational Parameters
    h0 = signal_params['h0']
    tstart = signal_params['tstart']
    Tobs = signal_params['Tobs']
    dt_wf = signal_params['dt_wf']
    detector = signal_params['detector']
    Tsft = signal_params['Tsft']
    sqrtSX = signal_params['sqrtSX']
    label = signal_params['label']
    save_path = signal_params['save_path']
    signal_idx = signal_params['signal_idx']
    
    freq, f1dot, f2dot, f3dot, f4dot = freq_params
    # Base spindown deviation over the observation time
    df_spindown = abs(f1dot) * Tobs
    df_glitch = 0.0
    
    # Convert relative parameters to absolute, and normalize tglitch relative to tref
    glitch_params_norm = []
    for gp in glitch_params:
        tglitch_rel, dnu_nu, dnu1_nu1, Q, tau = gp
        dnu_p, dnu_t, dnu1_p = calc_absolute_glitch_params(freq, f1dot, dnu_nu, dnu1_nu1, Q)
        glitch_params_norm.append([tglitch_rel - tref, dnu_p, dnu_t, dnu1_p, tau])
    
        df_glitch += abs(dnu_p) + abs(dnu_t) + (abs(dnu1_p) * Tobs)
        
    wf = waveform(h0, cosi, freq, f1dot, f2dot, f3dot, f4dot, glitch_params_norm)
    
    signal_out_dir = os.path.join(save_path, f"simCW{signal_idx}")
    temp_dir = os.path.join(save_path, 'tmp', f"simCW{signal_idx}")
    
    os.makedirs(signal_out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    S = simulateCW.CWSimulator(tref, tstart, Tobs, wf, dt_wf, phi0, psi, alpha, delta, detector)

    df_doppler = 1.05e-4 * freq
    safety_buffer = 0.01  # Small 0.01 Hz buffer for higher-order derivatives and binning
    df = df_spindown + df_glitch + df_doppler + safety_buffer

    fmin_nb = float(np.floor(freq - df))
    fmax_nb = float(np.ceil(freq + df))
    fband = int(fmax_nb-fmin_nb) 

    for file, j, N in S.write_sft_files(noise_sqrt_Sh=sqrtSX, fmax=fmax_nb, Tsft=Tsft, comment=f"simCW{signal_idx}", out_dir=temp_dir):
        pass
    
    combine_sfts(fmin=fmin_nb, fmax=fmax_nb, fband=fband, ts=tstart, te=tstart+Tobs, output=signal_out_dir, sft_dir=temp_dir)


def main(df, obs_params, save_path, n_cpu):
    # Group by signal index to handle multiple glitches per pulsar
    signals = df.groupby('n_th_signal')
    sim_args = []
    
    for signal_idx, group in signals:
        # Extract Astrophysical Parameters
        row = group.iloc[0]
        freq_params = [row['f0'], row['f1'], row['f2'], row['f3'], row['f4']]
        phi0, psi, cosi = row['phi0'], row['psi'], row['cosi']
        alpha, delta = row['alpha'], row['delta']
        
        # Glitch parameteres
        glitch_params = []
        for _, g_row in group.iterrows():
            # Check if tglitch exists and is not NaN (handles m=0 cases gracefully)
            if 'tglitch' in g_row and pd.notna(g_row['tglitch']):
                glitch_params.append([
                    g_row['tglitch'], g_row['dnu_nu'], g_row['dnu1_nu1'], g_row['Q'], g_row['tau']
                ])
        
        # 3. Combine with Observational Parameters
        sim_args.append({
            # Astrophysical
            'signal_idx': int(signal_idx),
            'freq_params': freq_params,
            'phi0': phi0, 'psi': psi, 'cosi': cosi,
            'alpha': alpha, 'delta': delta,
            'glitch_params': glitch_params,

            'h0': obs_params['h0'],
            'sqrtSX': obs_params['sqrtSX'],
            'tstart': obs_params['tstart'],
            'Tobs': obs_params['Tobs'],
            'tref': obs_params['tref'],
            'dt_wf': obs_params['dt_wf'],
            'detector': obs_params['detector'],
            'Tsft': obs_params['Tsft'],
            'label': obs_params['label'],
            'save_path': save_path
        })
        
    n_signals = len(sim_args)
    print(f"Loaded {n_signals} signals. Starting simulation pool with {n_cpu} CPUs...")
    
    t0 = time.time()
    with mp.Pool(processes=n_cpu) as pool:
        list(tqdm(pool.imap_unordered(simulate_signal, sim_args), total=n_signals, desc="Simulating signals"))
    
    temp_dir = os.path.join(save_path, 'tmp')
    shutil.rmtree(temp_dir)

    
    print(f"\nDone. Time used: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run continuous wave simulations from CSV parameters.")
    parser.add_argument('--n_cpu', type=int, default=8, help="Number of CPU cores to use")
    parser.add_argument('--label', default='wglitch_f1_3e-9', help="Label for data directory")
    parser.add_argument('--freq', type=float, default=100.0, help="Target frequency")
    
    args = parser.parse_args() 
    
    # Define Parameters 
    depth = 50
    sqrtSX = 1e-23 
    h0 = sqrtSX / depth 
    
    obs_params = {
        'freq': args.freq,
        'label': args.label,
        'depth': depth,
        'sqrtSX': sqrtSX,
        'h0': h0,
        'tstart': 1368970000,
        'Tobs': 80 * 86400,
        'tref': 1368970000 + 40 * 86400,
        'dt_wf': 5,
        'detector': 'H1',
        'Tsft': 1800
    }
    print(f"Observational Setup -> depth:{depth}, sqrtSX:{sqrtSX}, h0:{h0}")
    
    save_path = f'/scratch/kriles_root/kriles0/damoncht/simGlitch/data/{args.label}/100-100Hz'
    csv_path = f'{save_path}/signal_glitch_params.csv'
    print(f"Loading astrophysical parameters from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Did you run the generation script first?")
        
    df = pd.read_csv(csv_path)
    
    main(df, obs_params, save_path, n_cpu=args.n_cpu)