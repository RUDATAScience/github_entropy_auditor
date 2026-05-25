import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def run_advanced_proofs_and_export():
    """
    相転移の存在を証明する3つの高度な統計力学的指標を計算し、
    結果をCSVとグラフ画像としてZIP出力する。
    """
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Advanced Proofs of Phase Transitions in Opinion Dynamics', fontsize=16)

    csv_hk = 'hk_critical_slowing.csv'
    csv_galam = 'galam_pdf_heatmap.csv'
    csv_integrated = 'integrated_fluctuations.csv'
    plot_file = 'advanced_proofs_plot.png'
    zip_file = 'advanced_proofs.zip'

    # ==========================================
    # 1. HKモデル: 臨界減速 (Critical Slowing Down) の証明
    # 閾値付近で収束までの時間が発散（ピーク化）する現象を捉える
    # ==========================================
    print("1/3: HKモデルの「臨界減速」を計算中...")
    epsilons = np.linspace(0.05, 0.35, 30)
    hk_results = []
    N_hk = 200
    max_steps = 300
    tol = 1e-4

    for eps in epsilons:
        trials = 5
        avg_conv_time = 0
        for _ in range(trials):
            opinions = np.random.uniform(-1, 1, N_hk)
            conv_step = max_steps
            for step in range(max_steps):
                new_opinions = np.zeros(N_hk)
                for i in range(N_hk):
                    neighbors = opinions[np.abs(opinions - opinions[i]) <= eps]
                    new_opinions[i] = np.mean(neighbors)
                
                # 意見の変化量が閾値以下になれば収束とみなす
                if np.max(np.abs(new_opinions - opinions)) < tol:
                    conv_step = step
                    break
                opinions = new_opinions
            avg_conv_time += conv_step
        
        avg_conv_time /= trials
        hk_results.append({"Tolerance_Epsilon": eps, "Convergence_Time": avg_conv_time})

    df_hk = pd.DataFrame(hk_results)
    df_hk.to_csv(csv_hk, index=False)

    axes[0].plot(df_hk['Tolerance_Epsilon'], df_hk['Convergence_Time'], marker='o', color='blue')
    axes[0].axvline(x=0.2, color='red', linestyle='--', label=r'Critical $\epsilon_c \approx 0.2$')
    axes[0].set_title('1. HK Model: Critical Slowing Down')
    axes[0].set_xlabel(r'Tolerance $\epsilon$')
    axes[0].set_ylabel('Convergence Time (Steps)')
    axes[0].grid(True, ls="--", alpha=0.5)
    axes[0].legend()

    # ==========================================
    # 2. Galamモデル: 確率密度関数(PDF)のデルタ関数化の証明
    # Nの拡大に伴い分布が一点に収束する様子をヒートマップ化
    # ==========================================
    print("2/3: Galamモデルの「確率密度分布の集中」を計算中...")
    N_list = np.logspace(1, 6, num=12, base=10, dtype=float)
    galam_results = []
    J, H, beta = 1.0, 0.02, 1.0/1.2
    trials_galam = 200
    
    # ヒートマップ用の2次元配列
    bins = np.linspace(-1, 1, 40)
    pdf_matrix = np.zeros((len(N_list), len(bins)-1))

    for idx, N in enumerate(N_list):
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
            galam_results.append({"System_Size_N": N, "Final_Magnetization": m})
        
        hist, _ = np.histogram(final_m, bins=bins, density=True)
        pdf_matrix[idx, :] = hist

    df_galam = pd.DataFrame(galam_results)
    df_galam.to_csv(csv_galam, index=False)

    # 2Dヒートマップの描画
    sns.heatmap(pdf_matrix.T, cmap='viridis', ax=axes[1], cbar_kws={'label': 'Probability Density'},
                xticklabels=[f"10^{int(np.log10(n))}" for n in N_list], 
                yticklabels=np.round(bins[:-1], 2))
    axes[1].axvline(x=7, color='red', linestyle='--', label=r'Signal Cliff ($N_c = 10^4$)') # 10^4のインデックス付近
    axes[1].set_title('2. Galam Model: PDF Collapse to Delta Function')
    axes[1].set_xlabel('System Size $N$')
    axes[1].set_ylabel('Opinion Value (Magnetization)')
    axes[1].invert_yaxis()
    axes[1].legend()

    # ==========================================
    # 3. 統合モデル: ゆらぎ（感受率）の発散の証明
    # 閾値A_critにおいて、試行間の結果のばらつきが最大化する現象
    # ==========================================
    print("3/3: 統合モデルの「ゆらぎ（感受率）の発散」を計算中...")
    A_max_list = np.linspace(0.3, 0.9, 25)
    integrated_results = []
    N_int = 300
    dt = 0.1
    steps_int = 400
    trials_int = 15

    for A_max in A_max_list:
        centrist_ratios = np.zeros(trials_int)
        for trial in range(trials_int):
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
                
            centrist_ratios[trial] = np.mean(np.abs(I) < 0.25) * 100
        
        # 試行間の「分散（ゆらぎ）」を計算（これが相転移の感受率に相当する）
        var_centrist = np.var(centrist_ratios)
        integrated_results.append({"Max_Crisis_Level_A": A_max, "Centrist_Ratio_Variance": var_centrist})

    df_integrated = pd.DataFrame(integrated_results)
    df_integrated.to_csv(csv_integrated, index=False)

    axes[2].plot(df_integrated['Max_Crisis_Level_A'], df_integrated['Centrist_Ratio_Variance'], marker='^', color='orange')
    axes[2].axvline(x=0.6, color='red', linestyle='--', label=r'Plastic Threshold $A_{crit} = 0.6$')
    axes[2].set_title('3. Integrated Model: Divergence of Fluctuations')
    axes[2].set_xlabel('Maximum Crisis Level $A_{max}$')
    axes[2].set_ylabel('Variance of Centrist Ratio (Susceptibility)')
    axes[2].grid(True, ls="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_file, dpi=300)
    plt.show()

    # ==========================================
    # ZIP化とダウンロード
    # ==========================================
    print("データをZIPファイルに圧縮しています...")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_hk)
        zipf.write(csv_galam)
        zipf.write(csv_integrated)
        zipf.write(plot_file)
        
    print(f"圧縮完了: {zip_file}")
    try:
        from google.colab import files
        files.download(zip_file)
        print("ダウンロードを開始しました。")
    except ImportError:
        print(f"ローカル環境のため、カレントディレクトリに '{zip_file}' を保存しました。")

run_advanced_proofs_and_export()
