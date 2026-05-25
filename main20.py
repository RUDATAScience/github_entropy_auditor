import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def simulate_unanimity_limits():
    """
    1〜10の選択肢(M)において、集団サイズ(N)の全員が
    偶然完全に一致する確率の減衰と限界閾値を計算するシミュレーション。
    """
    # 選択肢の数 M: 1 から 10
    M_values = range(1, 11)
    
    # 集団サイズ N: 1 から 1200 まで（Float64限界を見届けるため）
    N_values = np.arange(1, 1201, 5) 
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    print("全員一致確率の限界シミュレーションを開始します...")
    
    generated_files = []
    
    for M in M_values:
        results = []
        plot_probs = []
        
        for N in N_values:
            if M == 1:
                # 選択肢が1つしかない場合は常に100%一致
                p = 1.0
                log10_p = 0.0
            else:
                # 全員が偶然同じ選択肢を選ぶ確率: M * (1/M)^N = (1/M)^(N-1)
                log10_p = (N - 1) * np.log10(1.0 / M)
                
                # Float64の限界 (-323.3) を下回る場合は完全な0.0に丸める
                if log10_p < -323:
                    p = 0.0
                else:
                    p = (1.0 / M) ** (N - 1)
            
            # プロット用（対数グラフで0を描画できないため、Float64の最小値でクリップ）
            safe_plot_p = p if p > 0.0 else 1e-323
            plot_probs.append(safe_plot_p)
            
            results.append({
                "Group_Size_N": N,
                "Choices_M": M,
                "Unanimity_Probability": p,
                "Log10_Probability": log10_p,
                "Is_Underflow_Zero": (p == 0.0)
            })
            
        # グラフへの描画
        ax.plot(N_values, plot_probs, color=colors[M-1], lw=2, label=f'Choices $M={M}$')
        
        # Mごとの結果を独立したCSVとして保存
        df = pd.DataFrame(results)
        csv_filename = f"unanimity_M{M}.csv"
        df.to_csv(csv_filename, index=False)
        generated_files.append(csv_filename)

    # ==========================================
    # グラフの装飾と限界閾値の描画
    # ==========================================
    ax.set_yscale('log')
    ax.set_ylim(1e-330, 10)
    ax.set_xlim(0, 1200)
    
    # 1. 社会科学的限界 (10^-9): 全人類に1回の偶然
    ax.axhline(y=1e-9, color='red', linestyle=':', lw=2, label=r'Social Limit ($10^{-9}$): 1 in a Billion')
    
    # 2. 物理的限界 (10^-80): 宇宙に存在する観測可能な原子の数
    ax.axhline(y=1e-80, color='purple', linestyle=':', lw=2, label=r'Physical Limit ($10^{-80}$): Atoms in Universe')
    
    # 3. 計算機的限界 (Float64アンダーフロー)
    ax.axhline(y=1e-308, color='black', linestyle='--', lw=2, label=r'Computational Limit ($10^{-308}$): Float64 Underflow')
    
    ax.set_title('Probability of Complete Unanimity by Chance (Choices M=1 to 10)', fontsize=16)
    ax.set_xlabel('Group Size $N$', fontsize=14)
    ax.set_ylabel('Probability of Unanimity (Log Scale)', fontsize=14)
    ax.legend(loc='upper right', ncol=2, fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plot_filename = "unanimity_limits_plot.png"
    plt.savefig(plot_filename, dpi=300)
    plt.show()
    generated_files.append(plot_filename)
    
    # ==========================================
    # ZIP化とダウンロード
    # ==========================================
    zip_filename = "unanimity_simulation.zip"
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
simulate_unanimity_limits()
