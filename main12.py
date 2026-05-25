import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
import warnings

warnings.filterwarnings('ignore')

def run_final_validations():
    """
    案2: GMMにおける浮動小数点アンダーフロー（事後分布の崩壊）
    案3: マイクロ・チャンキングによるマイノリティ救出（エルゴード性の意図的破壊）
    の2つのシミュレーションを実行し、結果を出力する。
    """
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Final Validation: Posterior Collapse vs. Micro-Chunking Restoration', fontsize=16)

    csv_underflow = 'gmm_underflow_results.csv'
    csv_chunking = 'micro_chunking_results.csv'
    plot_file = 'final_validation_plot.png'
    zip_file = 'final_validation.zip'

    # ==========================================
    # 案2: 浮動小数点アンダーフローによる「事後分布の完全な崩壊」
    # Nが大きくなると、マイノリティ仮説の事後確率がFloat64の限界を超えて「完全なゼロ(0.0)」になる現象
    # ==========================================
    print("案2: 浮動小数点アンダーフローのシミュレーションを実行中...")
    N_list = np.linspace(10, 1000, 100, dtype=int)
    underflow_results = []
    
    # マイノリティがマジョリティモデルから外れる度合い（KLダイバージェンス相当の定数）
    kl_divergence_penalty = 1.0 

    for N in N_list:
        # ベイズ推論におけるマイノリティ仮説の事後確率ウェイト W_minority ∝ exp(-N * KL)
        # 対数空間ではなく、実際の確率空間(0.0 ~ 1.0)でアルゴリズムが評価する際の生の値を計算
        raw_probability = np.exp(-N * kl_divergence_penalty)
        
        underflow_results.append({
            "System_Size_N": N,
            "Minority_Posterior_Probability": raw_probability,
            "Is_Underflow_Zero": raw_probability == 0.0
        })

    df_underflow = pd.DataFrame(underflow_results)
    df_underflow.to_csv(csv_underflow, index=False)

    axes[0].plot(df_underflow['System_Size_N'], df_underflow['Minority_Posterior_Probability'], color='purple', lw=2)
    # Float64がアンダーフローを起こす約 N=745 付近に線を引く
    axes[0].axvline(x=745, color='red', linestyle='--', label='Float64 Underflow Limit ($N \\approx 745$)')
    axes[0].fill_between(df_underflow['System_Size_N'], df_underflow['Minority_Posterior_Probability'], 0, alpha=0.3, color='purple')
    axes[0].set_title('Idea 2: Posterior Collapse via Float64 Underflow')
    axes[0].set_xlabel('System Size $N$')
    axes[0].set_ylabel('Minority Posterior Probability (Raw Scale)')
    axes[0].grid(True, ls="--", alpha=0.5)
    axes[0].legend()

    # ==========================================
    # 案3: マイクロ・チャンキングによるマイノリティの救出
    # 100万行のデータを一括処理するか、分割処理するかでの生存クラスター数の比較
    # ==========================================
    print("案3: マイクロ・チャンキングのシミュレーションを実行中...")
    num_clusters = 20
    alpha = 1.2
    ranks = np.arange(1, num_clusters + 1)
    p_true = 1.0 / (ranks ** alpha)
    p_true /= np.sum(p_true)  # ジップの法則に基づく20クラスターの真の比率
    
    threshold_ratio = 0.02  # 検出閾値 2.0%
    N_total = 1000000       # 100万行
    
    # チャンクサイズ（バッチサイズ）のリスト
    chunk_sizes = np.logspace(3, 6, 20, dtype=int)
    chunking_results = []

    for N_chunk in chunk_sizes:
        num_batches = max(1, N_total // N_chunk)
        surviving_clusters = 0
        
        for pk in p_true:
            k_thresh = max(0, int(np.ceil(N_chunk * threshold_ratio)) - 1)
            # 各チャンクにおいて、マイノリティが偶然閾値（2%）を超える確率
            prob_detected_in_chunk = binom.sf(k_thresh, N_chunk, pk)
            
            # 全バッチを通じて1回でも検出されれば生存とみなす
            if prob_detected_in_chunk < 1e-15:
                prob_survive_global = 0.0
            else:
                prob_survive_global = 1.0 - (1.0 - prob_detected_in_chunk)**num_batches
                
            if prob_survive_global > 0.5:
                surviving_clusters += 1
                
        chunking_results.append({
            "Chunk_Size_N": N_chunk,
            "Num_Batches": num_batches,
            "Surviving_Clusters": surviving_clusters
        })

    df_chunking = pd.DataFrame(chunking_results)
    df_chunking.to_csv(csv_chunking, index=False)

    axes[1].plot(df_chunking['Chunk_Size_N'], df_chunking['Surviving_Clusters'], marker='o', color='green', lw=2)
    axes[1].axvline(x=10000, color='red', linestyle='--', label='Optimal Chunk Limit ($N_c \\approx 10^4$)')
    axes[1].set_xscale('log')
    axes[1].set_title('Idea 3: Restoring Ergodicity via Micro-Chunking')
    axes[1].set_xlabel('Chunk Size $N_{chunk}$ (Log Scale)')
    axes[1].set_ylabel('Number of Surviving Clusters (Out of 20)')
    axes[1].set_ylim(0, 21)
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_file, dpi=300)
    plt.show()

    # ==========================================
    # ZIP化とダウンロード
    # ==========================================
    print("\nデータをZIPファイルに圧縮しています...")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_underflow)
        zipf.write(csv_chunking)
        zipf.write(plot_file)
        
    print(f"圧縮完了: {zip_file}")
    try:
        from google.colab import files
        files.download(zip_file)
        print("ダウンロードを開始しました。")
    except ImportError:
        print(f"ローカル環境のため、カレントディレクトリに '{zip_file}' を保存しました。")

run_final_validations()
