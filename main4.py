import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, levene
import warnings

# 警告の非表示（正規近似時の微小な計算誤差ワーニング等を防ぐため）
warnings.filterwarnings('ignore')

def simulate_galam_with_statistics():
    """
    Galamモデルに基づくオピニオンダイナミクス・シミュレーション
    Signal Cliffの統計学的有意性検定とCSV出力、可視化を行う
    """
    # --- 1. パラメータ設定 ---
    J = 1.0          # 同調圧力 (Conformity pressure)
    H = 0.02         # 外部バイアス (Systemic bias)
    T = 1.2          # 社会的温度 (Social temperature)
    beta = 1.0 / T
    
    time_steps = 50  # 収束までの時間ステップ
    num_trials = 500 # 各システムサイズ N における試行回数

    # 検証するシステムサイズ N のリスト (10^1 から 10^10)
    N_list = np.logspace(1, 10, num=10, base=10, dtype=float)
    
    results_data = [] # CSV出力用のリスト
    variances = []
    entropies = []
    
    print("シミュレーションを開始します...")

    # --- 2. メインループ: 各システムサイズ N について計算 ---
    for N in N_list:
        final_magnetizations = np.zeros(num_trials)
        
        for trial in range(num_trials):
            # 初期磁化 (m): ほぼ中立 (0) だが、わずかなゆらぎを持たせる
            m = np.random.normal(0, 0.05)
            
            for t in range(time_steps):
                p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * (J * m + H)))
                
                if N < 1e6:
                    n_plus = np.random.binomial(n=int(N), p=p_plus)
                else:
                    mu = N * p_plus
                    sigma = np.sqrt(N * p_plus * (1.0 - p_plus))
                    n_plus = np.random.normal(loc=mu, scale=sigma)
                    n_plus = np.clip(n_plus, 0, N)
                
                m = (2.0 * n_plus - N) / N
                
            final_magnetizations[trial] = m
            
            # データの記録
            results_data.append({
                "Population_N": N,
                "Log10_N": int(np.log10(N)),
                "Trial_ID": trial + 1,
                "Final_Magnetization": m
            })
            
        var_m = np.var(final_magnetizations)
        variances.append(var_m)
        
        hist, bin_edges = np.histogram(final_magnetizations, bins=50, density=True)
        hist_prob = hist[hist > 0] * np.diff(bin_edges)[hist > 0]
        S = entropy(hist_prob, base=2)
        entropies.append(S)

    # --- 3. データフレーム化とCSV保存 ---
    df = pd.DataFrame(results_data)
    csv_filename = "galam_signal_cliff_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n全試行のデータを '{csv_filename}' に保存しました。")

    # --- 4. 統計学的有意性の検定 (Levene検定) ---
    # Signal Cliff (N=10^4) の前後である N=10^3 と N=10^5 で「分散に差があるか」を検定する
    m_10e3 = df[df["Log10_N"] == 3]["Final_Magnetization"]
    m_10e5 = df[df["Log10_N"] == 5]["Final_Magnetization"]
    
    # Levene検定 (正規分布に従わないデータに対してもロバストな等分散性検定)
    stat, p_val = levene(m_10e3, m_10e5)
    
    print("\n--- 統計解析結果 (分散の崩壊の証明) ---")
    print(f"N = 10^3 (Cliff前) の分散: {np.var(m_10e3):.2e}")
    print(f"N = 10^5 (Cliff後) の分散: {np.var(m_10e5):.2e}")
    print(f"Levene検定 p値: {p_val:.2e}")
    if p_val < 0.01:
        print("=> p < 0.01 です。Nの拡大に伴う「分散の崩壊（Signal Cliff）」は統計学的に極めて有意です。")

    # --- 5. 結果の可視化 (プロット) ---
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Validation of Signal Cliff & Epistemic Injustice (Galam Model)', fontsize=16)

    # 上段: バイオリンプロット（分布の崩壊を直接可視化）
    ax1 = plt.subplot(2, 1, 1)
    sns.violinplot(x="Log10_N", y="Final_Magnetization", data=df, ax=ax1, palette="viridis", inner="quartile")
    ax1.set_title(f'Probability Density of Final States (Levene Test p-value: {p_val:.2e})')
    ax1.set_xlabel('Population Size ($10^x$)')
    ax1.set_ylabel('Final Magnetization $m$')
    ax1.axvline(x=3.5, color='red', linestyle='--', alpha=0.7, label='Signal Cliff Threshold')
    ax1.legend()

    # 下段左: 分散の崩壊
    ax2 = plt.subplot(2, 2, 3)
    ax2.plot(N_list, variances, marker='o', color='purple', linestyle='-')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.axvline(x=1e4, color='red', linestyle='--', alpha=0.7, label='Signal Cliff ($N=10^4$)')
    ax2.set_title('Variance Collapse: The "Signal Cliff"')
    ax2.set_xlabel('Population Size ($N$)')
    ax2.set_ylabel('Variance (log scale)')
    ax2.legend()

    # 下段右: エントロピーの崩壊
    ax3 = plt.subplot(2, 2, 4)
    ax3.plot(N_list, entropies, marker='^', color='green', linestyle='-')
    ax3.set_xscale('log')
    ax3.axvline(x=1e4, color='red', linestyle='--', alpha=0.7, label='Signal Cliff ($N=10^4$)')
    ax3.set_title('Shannon Entropy Collapse')
    ax3.set_xlabel('Population Size ($N$)')
    ax3.set_ylabel('Information Entropy (bits)')
    ax3.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 関数の実行
simulate_galam_with_statistics()
