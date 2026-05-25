import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom, entropy
import warnings

warnings.filterwarnings('ignore')

def simulate_cluster_survival_in_big_data():
    """
    1000万行のSNSデータ（20の意見クラスター）における、
    分析ウィンドウサイズNとマイノリティクラスターの生存確率のシミュレーション
    """
    np.random.seed(42)
    
    # 1. データセットの設定
    N_total = 10000000  # 全データ数: 1000万行
    num_clusters = 20   # クラスターの数
    alpha = 1.2         # ジップの法則（べき乗則）の指数
    
    # べき乗則に従う真の確率分布 P(k)
    ranks = np.arange(1, num_clusters + 1)
    p_true = 1.0 / (ranks ** alpha)
    p_true /= np.sum(p_true)  # 確率の正規化
    
    # クラスタリングアルゴリズムのノイズ閾値
    # (バッチ内で2.5%以上の割合を持たないとノイズとして上位クラスターに吸収されると仮定)
    threshold_ratio = 0.025
    
    # 検証する分析ウィンドウサイズ（バッチサイズ）N: 100件 〜 1000万件
    N_windows = np.logspace(2, 7, num=25, base=10, dtype=int)
    
    results = []
    
    print("1000万行のデータ分割シミュレーションを開始します...")
    
    # 2. メインループ: 各分析ウィンドウサイズ N でのシミュレーション
    for N in N_windows:
        num_batches = max(1, N_total // N)
        surviving_clusters_expected = 0
        surviving_probs = []
        
        for k in range(num_clusters):
            pk = p_true[k]
            # 各バッチでクラスターkが検出閾値を超える確率 (二項分布の生存関数)
            k_thresh = max(0, int(np.ceil(N * threshold_ratio)) - 1)
            prob_detected_in_one_batch = binom.sf(k_thresh, N, pk)
            
            # 全バッチの中で、少なくとも1つのバッチで検出される確率
            if prob_detected_in_one_batch < 1e-15:
                prob_survive_global = 0.0
            else:
                prob_survive_global = 1.0 - (1.0 - prob_detected_in_one_batch)**num_batches
                
            surviving_clusters_expected += prob_survive_global
            
            # 確率が50%以上なら、マクロな最終結果として生存したとみなす
            if prob_survive_global > 0.5:
                surviving_probs.append(pk)
                
        # 生存したクラスターのみでシャノンエントロピー（多様性）を計算
        if len(surviving_probs) > 0:
            surviving_probs = np.array(surviving_probs) / np.sum(surviving_probs)
            S = entropy(surviving_probs, base=2)
        else:
            S = 0
            
        results.append({
            "Window_Size_N": N,
            "Num_Batches": num_batches,
            "Expected_Surviving_Clusters": surviving_clusters_expected,
            "Shannon_Entropy_bits": S
        })
        print(f"バッチサイズ N={N:<8} | バッチ数={num_batches:<6} | 生存クラスター数={surviving_clusters_expected:.1f}/20 | エントロピー={S:.2f}")

    df = pd.DataFrame(results)
    csv_file = "clustering_survival_results.csv"
    df.to_csv(csv_file, index=False)
    
    # 3. グラフの可視化
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color1 = '#2980b9'
    ax1.plot(df['Window_Size_N'], df['Expected_Surviving_Clusters'], marker='o', color=color1, linewidth=2, label='Surviving Clusters (out of 20)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Analysis Window Size $N_{window}$ (Log Scale)', fontsize=14)
    ax1.set_ylabel('Expected Number of Surviving Clusters', color=color1, fontsize=14)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Signal Cliff (N = 10^4) 付近に警告線を描画
    ax1.axvline(x=1e4, color='#e74c3c', linestyle='--', linewidth=2, label=r'Signal Cliff ($N_c \approx 10^4$)')
    
    # エントロピーのグラフを第2軸に描画
    ax2 = ax1.twinx()
    color2 = '#27ae60'
    ax2.plot(df['Window_Size_N'], df['Shannon_Entropy_bits'], marker='s', color=color2, linewidth=2, linestyle='-.', label='Shannon Entropy (Diversity)')
    ax2.set_ylabel('Shannon Entropy (bits)', color=color2, fontsize=14)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig.suptitle('Algorithmic Absorption in 10M Data: The Optimal Chunking Threshold', fontsize=16)
    
    # 背景のフェーズ色分け
    ax1.axvspan(100, 1e4, color='green', alpha=0.05, label='Stochastic Phase (Micro-chunking)')
    ax1.axvspan(1e4, 1e7, color='red', alpha=0.05, label='Deterministic Phase (Algorithmic Tyranny)')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left', fontsize=12)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plot_file = "clustering_survival_plot.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.show()
    
    # 4. ZIP化とダウンロード
    zip_file = "sns_clustering_simulation.zip"
    print("\nデータをZIPファイルに圧縮しています...")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_file)
        zipf.write(plot_file)
        
    print(f"圧縮完了: {zip_file}")
    try:
        from google.colab import files
        files.download(zip_file)
        print("ダウンロードを開始しました。")
    except ImportError:
        print(f"ローカル環境のため、カレントディレクトリに '{zip_file}' を保存しました。")

# 実行
simulate_cluster_survival_in_big_data()
