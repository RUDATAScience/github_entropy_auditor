import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

def simulate_trillion_scale_nlp_clustering():
    """
    Simulate the phase transition of minority pattern survival based on 
    total data size (N_total) and analysis window size (N_window) 
    in massive scale NLP corpora (1 Billion to 1 Trillion).
    """
    np.random.seed(42)
    
    # Dataset settings (1 Billion, 10 Billion, 100 Billion, 1 Trillion)
    N_totals = [10**9, 10**10, 10**11, 10**12]
    labels = ['1B', '10B', '100B', '1T (Trillion)']
    
    # NLP context: Many rare linguistic patterns (deep long tail)
    # Expanding to 500 clusters to verify extreme semantic minorities
    num_clusters = 500 
    alpha = 1.15 # Strong Zipf's law typical in natural languages
    threshold_ratio = 0.01 # 1% detection threshold within a chunk
    
    ranks = np.arange(1, num_clusters + 1)
    p_true = 1.0 / (ranks ** alpha)
    p_true /= np.sum(p_true)
    
    # Window sizes from 10^2 up to 10^8
    N_windows = np.logspace(2, 8, num=40, base=10, dtype=int)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#f39c12', '#e74c3c', '#8e44ad', '#2c3e50']
    
    print("Starting Trillion-scale NLP corpus partition simulation...")
    
    generated_files = [] 
    
    for idx, N_total in enumerate(N_totals):
        surviving_counts = []
        current_scale_results = []
        
        for N in N_windows:
            num_batches = max(1, N_total // N)
            surviving_clusters_expected = 0
            
            for k in range(num_clusters):
                pk = p_true[k]
                k_thresh = max(0, int(np.ceil(N * threshold_ratio)) - 1)
                
                # Using Normal approximation for Binomial to prevent overflow at massive N
                mean = N * pk
                var = N * pk * (1.0 - pk)
                if var > 0:
                    z = (k_thresh - mean) / np.sqrt(var)
                    p_det = norm.sf(z)
                else:
                    p_det = 1.0 if mean > k_thresh else 0.0
                
                # Prevent floating point underflow
                if p_det < 1e-15:
                    prob_survive = 0.0
                else:
                    prob_survive = 1.0 - np.exp(-num_batches * p_det)
                    
                surviving_clusters_expected += prob_survive
                
            surviving_counts.append(surviving_clusters_expected)
            current_scale_results.append({
                "Total_Data_Scale": f"10^{int(np.log10(N_total))}",
                "Window_Size_N": N,
                "Num_Batches": num_batches,
                "Expected_Surviving_Patterns": surviving_clusters_expected
            })
            
        ax.plot(N_windows, surviving_counts, marker='o', markersize=4, 
                color=colors[idx], linewidth=2, label=f'Total Corpus: {labels[idx]}')

        # Save results for this specific scale as a separate CSV
        df = pd.DataFrame(current_scale_results)
        csv_filename = f"nlp_survival_{labels[idx].split(' ')[0]}.csv"
        df.to_csv(csv_filename, index=False)
        generated_files.append(csv_filename)
    
    # Graph formatting
    ax.set_xscale('log')
    ax.set_xlabel('Analysis Window Size (Chunk Size) $N_{window}$', fontsize=14)
    ax.set_ylabel(f'Surviving NLP Patterns (out of {num_clusters})', fontsize=14)
    ax.axvline(x=1e4, color='black', linestyle='--', linewidth=2, label=r'Signal Cliff ($N_c \approx 10^4$)')
    ax.set_title('NLP Pattern Survival across Trillion-Scale Corpora', fontsize=16)
    ax.legend(loc='lower left', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plot_filename = "trillion_scale_survival_plot.png"
    plt.savefig(plot_filename, dpi=300)
    plt.show()
    generated_files.append(plot_filename)
    
    # ==========================================
    # Zipping and Downloading Process
    # ==========================================
    zip_filename = "trillion_scale_nlp_simulation.zip"
    print("\nCompressing data into ZIP file...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in generated_files:
            zipf.write(file)
            
    print(f"Compression complete: {zip_filename}")
    
    # Automatic download for Google Colab environments
    try:
        from google.colab import files
        files.download(zip_filename)
        print("Download started.")
    except ImportError:
        print(f"Local environment detected. '{zip_filename}' has been saved.")

# Execute the simulation
simulate_trillion_scale_nlp_clustering()
