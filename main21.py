import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def kl_divergence(q, p):
    """
    2項分布におけるカルバック・ライブラー情報量 (Kullback-Leibler divergence)
    q: 実際の得票率 (観測された確率)
    p: ランダムな選択における期待値 (1/M)
    """
    # 0や1の極端なケースのエラー回避
    q = np.clip(q, 1e-15, 1.0 - 1e-15)
    return q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))

def simulate_electoral_anomaly():
    """
    ある選挙区において、当選者の得票数が「自然な揺らぎ」か
    「異常な圧力・不正」によるものかを 판別するシミュレーション。
    """
    np.random.seed(42)
    
    # 基準となる選挙区の投票人数 N (例: 10万人)
    N = 100000 
    
    # 候補者数 M のバリエーション
    M_values = [2, 3, 4, 5, 10]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(M_values)))
    
    print("選挙パフォーマンスの異常検知シミュレーションを開始します...")
    
    generated_files = []
    
    for idx, M in enumerate(M_values):
        results = []
        plot_probs = []
        
        # ランダムな場合の期待得票率 p = 1/M
        p_null = 1.0 / M
        
        # 得票率 q を ランダム期待値 から 100% まで変化させる
        q_values = np.linspace(p_null, 1.0, 200)
        
        for q in q_values:
            # 得票数 k
            k = int(N * q)
            
            # 大偏差原理に基づく確率の対数近似
            # Log10_P ≈ -N * D_KL(q || p) / ln(10)
            d_kl = kl_divergence(q, p_null)
            log10_p = -N * d_kl / np.log(10)
            
            # グラフ描画用 (Float64限界でクリップ)
            safe_plot_log10_p = max(log10_p, -350)
            plot_probs.append(safe_plot_log10_p)
            
            results.append({
                "Total_Voters_N": N,
                "Candidates_M": M,
                "Winner_Votes_k": k,
                "Vote_Share_q": q,
                "Log10_Probability": log10_p,
                "Is_Physically_Impossible": (log10_p < -80)
            })
            
        ax.plot(q_values * 100, plot_probs, color=colors[idx], lw=2, label=f'Candidates $M={M}$ (Expected: {p_null*100:.1f}%)')
        
        # Mごとの結果をCSVとして保存
        df = pd.DataFrame(results)
        csv_filename = f"electoral_anomaly_M{M}.csv"
        df.to_csv(csv_filename, index=False)
        generated_files.append(csv_filename)

    # ==========================================
    # グラフの装飾と限界閾値の描画
    # ==========================================
    ax.set_xlim(0, 100)
    ax.set_ylim(-330, 10)
    
    # 1. 社会科学的限界 (10^-9): 同調圧力・組織票の疑いが極めて濃厚
    ax.axhline(y=-9, color='red', linestyle=':', lw=2, label=r'Social Limit ($10^{-9}$): Strong Pressure')
    
    # 2. 物理的限界 (10^-80): 物理的に起こり得ない（不正の可能性大）
    ax.axhline(y=-80, color='purple', linestyle=':', lw=2, label=r'Physical Limit ($10^{-80}$): Impossible Bias')
    
    # 3. 計算機的限界 (Float64アンダーフロー)
    ax.axhline(y=-308, color='black', linestyle='--', lw=2, label=r'Computational Limit ($10^{-308}$)')
    
    ax.set_title(f'Electoral Anomaly Detection: Probability Decay vs Vote Share (Total Voters N={N:,})', fontsize=16)
    ax.set_xlabel('Winner\'s Vote Share (%)', fontsize=14)
    ax.set_ylabel('Log10 Probability of Occurrence by Chance', fontsize=14)
    ax.legend(loc='upper right', ncol=2, fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plot_filename = "electoral_anomaly_plot.png"
    plt.savefig(plot_filename, dpi=300)
    plt.show()
    generated_files.append(plot_filename)
    
    # ==========================================
    # ZIP化とダウンロード
    # ==========================================
    zip_filename = "electoral_anomaly_simulation.zip"
    print("\nデータをZIPファイルに圧縮しています...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in generated_files:
            zipf.write(file)
            
    print(f"圧縮完了: {zip_filename}")
    
    try:
        from google.colab import files
        files.download(zip_filename)
        print("ダウンロードを開始しました。")
    except ImportError:
        print(f"ローカル環境のため、カレントディレクトリに '{zip_filename}' を保存しました。")

# 実行
simulate_electoral_anomaly()
