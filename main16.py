import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
import warnings

warnings.filterwarnings('ignore')

def simulate_massive_scale_survival():
    """
    超大規模データ（10M, 100M, 1B, 10B）における
    データ総量 N_total と分析ウィンドウサイズ N_window に伴う
    マイノリティ・クラスター生存数の相転移をシミュレーションする。
    """
    np.random.seed(42)
    
    # データセットの設定 (1千万, 1億, 10億, 100億)
    N_totals = [10**7, 10**8, 10**9, 10**10]
    labels = ['10M', '100M', '1B', '10B']
    
    # ロングテールの限界を検証するためクラスター数を50に拡張
    num_clusters = 50 
    alpha = 1.2
    threshold_ratio = 0.025 # 2.5%の検出閾値
    
    ranks = np.arange(1, num_clusters + 1)
    p_true = 1.0 / (ranks ** alpha)
    p_true /= np.sum(p_true)
    
    N_windows = np.logspace(2, 7, num=30, base=10, dtype=int)
    
    results = []
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']
    
    for idx, N_total in enumerate(N_totals):
        surviving_counts = []
        for N in N_windows:
            num_batches = max(1, N_total // N)
            surviving_clusters_expected = 0
            
            for k in range(num_clusters):
                pk = p_true[k]
                k_thresh = max(0, int(np.ceil(N * threshold_ratio)) - 1)
                p_det = binom.sf(k_thresh, N, pk)
                
                # 大規模バッチの数値計算の安定化: (1 - p)^n \approx exp(-np)
                if p_det < 1e-15:
                    prob_survive = 0.0
                else:
                    prob_survive = 1.0 - np.exp(-num_batches * p_det)
                    
                surviving_clusters_expected += prob_survive
                
            surviving_counts.append(surviving_clusters_expected)
            results.append({
                "Total_Data_Scale": f"10^{int(np.log10(N_total))}",
                "Window_Size_N": N,
                "Num_Batches": num_batches,
                "Expected_Surviving_Clusters": surviving_clusters_expected
            })
            
        ax.plot(N_windows, surviving_counts, marker='o', markersize=5, 
                color=colors[idx], linewidth=2, label=f'Total Data: {labels[idx]}')

    df = pd.DataFrame(results)
    df.to_csv("massive_scale_survival_results.csv", index=False)
    
    ax.set_xscale('log')
    ax.set_xlabel('Analysis Window Size $N_{window}$ (Log Scale)', fontsize=14)
    ax.set_ylabel(f'Expected Surviving Clusters (out of {num_clusters})', fontsize=14)
    ax.axvline(x=1e4, color='black', linestyle='--', linewidth=2, label=r'Signal Cliff ($N_c \approx 10^4$)')
    ax.set_title('Cluster Survival across Massive Data Scales (Up to 10 Billion)', fontsize=16)
    ax.legend(loc='lower left', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("massive_scale_survival_plot.png", dpi=300)
    plt.show()

# 実行
simulate_massive_scale_survival()
