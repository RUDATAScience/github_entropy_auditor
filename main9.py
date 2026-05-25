import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import warnings

# 警告の非表示
warnings.filterwarnings('ignore')

def run_enhanced_simulations_and_export():
    """
    3つのオピニオンダイナミクスモデルの臨界閾値を計算し、
    追加検証指標とともにCSVおよびグラフ画像としてZIP出力する
    """
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Critical Thresholds in Three Different Opinion Dynamics Models', fontsize=16)

    # 保存用ファイル名の定義
    csv_hk = 'hk_model_results.csv'
    csv_galam = 'galam_model_results.csv'
    csv_integrated = 'integrated_model_results.csv'
    plot_file = 'thresholds_plot.png'
    zip_file = 'simulation_results.zip'

    # ==========================================
    # 1. Hegselmann-Krause (HK) モデル: 寛容度 ε の閾値検証
    # ==========================================
    print("1/3: HKモデルの寛容度閾値(ε)を計算中...")
    epsilons = np.linspace(0.05, 0.35, 30)
    hk_results = []
    N_hk = 200
    steps_hk = 50

    for eps in epsilons:
        opinions = np.random.uniform(-1, 1, N_hk)
        for _ in range(steps_hk):
            new_opinions = np.zeros(N_hk)
            for i in range(N_hk):
                neighbors = opinions[np.abs(opinions - opinions[i]) <= eps]
                new_opinions[i] = np.mean(neighbors)
            opinions = new_opinions
        
        # 評価指標の計算
        rounded_opinions = np.round(opinions, 2)
        unique_vals, counts = np.unique(rounded_opinions, return_counts=True)
        unique_clusters = len(unique_vals)
        largest_cluster_ratio = (np.max(counts) / N_hk) * 100 # 最大クラスタの人口割合
        
        hk_results.append({
            "Tolerance_Epsilon": eps,
            "Surviving_Clusters": unique_clusters,
            "Largest_Cluster_Ratio_pct": largest_cluster_ratio
        })

    df_hk = pd.DataFrame(hk_results)
    df_hk.to_csv(csv_hk, index=False)

    axes[0].plot(df_hk['Tolerance_Epsilon'], df_hk['Surviving_Clusters'], marker='o', color='blue', label='Clusters')
    axes[0].axvline(x=0.2, color='red', linestyle='--', label=r'Critical $\epsilon_c \approx 0.2$')
    axes[0].set_title('1. HK Model: Tolerance Threshold ($\epsilon$)')
    axes[0].set_xlabel(r'Tolerance $\epsilon$')
    axes[0].set_ylabel('Number of Surviving Clusters')
    axes[0].grid(True, ls="--", alpha=0.5)
    axes[0].legend()

    # ==========================================
    # 2. Galamモデル: システムサイズ N の閾値検証 (Signal Cliff)
    # ==========================================
    print("2/3: Galamモデルのシステムサイズ閾値(N_c)を計算中...")
    N_list = np.logspace(1, 8, num=15, base=10, dtype=float)
    galam_results = []
    J, H, beta = 1.0, 0.02, 1.0/1.2
    trials_galam = 100

    for N in N_list:
        final_m = np.zeros(trials_galam)
        for trial in range(trials_galam):
            m = np.random.normal(0, 0.05)
            for _ in range(30):
                p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * (J * m + H)))
                if N < 1e5:
                    n_plus = np.random.binomial(n=int(N), p=p_plus)
                else:
                    mu = N * p_plus
                    sig = np.sqrt(N * p_plus * (1.0 - p_plus))
                    n_plus = np.clip(np.random.normal(mu, sig), 0, N)
                m = (2.0 * n_plus - N) / N
            final_m[trial] = m
            
        var_m = np.var(final_m)
        
        # シャノンエントロピーの追加検証
        hist, bin_edges = np.histogram(final_m, bins=50, density=True)
        hist_prob = hist[hist > 0] * np.diff(bin_edges)[hist > 0]
        S = entropy(hist_prob, base=2) if len(hist_prob) > 0 else 0
        
        galam_results.append({
            "System_Size_N": N,
            "Log10_N": np.log10(N),
            "Opinion_Variance": var_m,
            "Shannon_Entropy_bits": S
        })

    df_galam = pd.DataFrame(galam_results)
    df_galam.to_csv(csv_galam, index=False)

    axes[1].plot(df_galam['System_Size_N'], df_galam['Opinion_Variance'], marker='s', color='purple')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].axvline(x=1e4, color='red', linestyle='--', label=r'Signal Cliff $N_c \approx 10^4$')
    axes[1].set_title('2. Galam Model: Scale Threshold ($N$)')
    axes[1].set_xlabel('System Size $N$ (Log Scale)')
    axes[1].set_ylabel('Variance of Opinion (Log Scale)')
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].legend()

    # ==========================================
    # 3. 統合的拡張モデル: 危機レベル A の閾値検証 (Plastic Deformation)
    # ==========================================
    print("3/3: 統合モデルの危機レベル閾値(A_crit)を計算中...")
    A_max_list = np.linspace(0.0, 1.0, 20)
    integrated_results = []
    N_int = 500
    dt = 0.1
    steps_int = 500

    for A_max in A_max_list:
        B = np.concatenate([np.random.normal(-0.8, 0.1, N_int//2), 
                            np.random.normal(0.8, 0.1, N_int//2)])
        I = np.copy(B)
        A_crit = 0.6
        gamma_plastic = 0.02
        
        for step in range(steps_int):
            curr_A = A_max if (100 < step < 300) else 0.0
            if curr_A > A_crit:
                B += gamma_plastic * (curr_A - A_crit) * (0.0 - B) * dt
                
            W_C = 0.3 + 4.5 * curr_A
            force_C = W_C * np.exp(-(I**2) / (2 * 0.4**2)) * (0.0 - I)
            force_B = 1.0 * (B - I)
            dI = (force_C + force_B) * dt + np.random.normal(0, 0.05, N_int) * np.sqrt(dt)
            I = np.clip(I + dI, -1.0, 1.0)
            
        centrist_ratio = np.mean(np.abs(I) < 0.25) * 100
        # 追加検証：極端派（|I| > 0.6）の残留割合
        extremist_ratio = np.mean(np.abs(I) > 0.6) * 100
        
        integrated_results.append({
            "Max_Crisis_Level_A": A_max,
            "Final_Centrist_Ratio_pct": centrist_ratio,
            "Final_Extremist_Ratio_pct": extremist_ratio
        })

    df_integrated = pd.DataFrame(integrated_results)
    df_integrated.to_csv(csv_integrated, index=False)

    axes[2].plot(df_integrated['Max_Crisis_Level_A'], df_integrated['Final_Centrist_Ratio_pct'], marker='^', color='orange', label='Centrist Ratio')
    # 極端派のグラフも追加でうすく描画
    axes[2].plot(df_integrated['Max_Crisis_Level_A'], df_integrated['Final_Extremist_Ratio_pct'], marker='v', color='gray', alpha=0.5, label='Extremist Ratio')
    axes[2].axvline(x=0.6, color='red', linestyle='--', label=r'Plastic Threshold $A_{crit} = 0.6$')
    axes[2].set_title('3. Integrated Model: Crisis Threshold ($A$)')
    axes[2].set_xlabel('Maximum Crisis Level $A_{max}$')
    axes[2].set_ylabel('Population Ratio (%)')
    axes[2].grid(True, ls="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_file, dpi=300) # 高画質で保存
    plt.show()

    # ==========================================
    # ZIP化とダウンロード処理
    # ==========================================
    print("データをZIPファイルに圧縮しています...")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_hk)
        zipf.write(csv_galam)
        zipf.write(csv_integrated)
        zipf.write(plot_file)
        
    print(f"圧縮完了: {zip_file}")

    # Colab環境での自動ダウンロード処理
    try:
        from google.colab import files
        files.download(zip_file)
        print("ダウンロードを開始しました。")
    except ImportError:
        print(f"ローカル環境のため、カレントディレクトリに '{zip_file}' を保存しました。")

# 実行
run_enhanced_simulations_and_export()
