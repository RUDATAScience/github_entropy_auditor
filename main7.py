import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levene
import warnings

# 警告の非表示（マクロ近似時の微小エラー回避）
warnings.filterwarnings('ignore')

def simulate_appendix_a_signal_cliff():
    """
    Appendix A: 巨大スケールにおける臨界閾値 Nc の計算実験
    シミュレーション実測値と理論的スケーリング則 (1/N) の比較
    """
    # --- 1. パラメータ設定 (Appendix A に準拠) ---
    J = 1.0          # 同調圧力
    H = 0.02         # 外部バイアス（微小な誘導力）
    T = 1.2          # 社会的温度
    beta = 1.0 / T
    
    time_steps = 50  # 収束までのステップ数
    num_trials = 500 # 各Nにおける試行回数（並行社会の数）

    # 検証するシステムサイズ N (10^1 から 10^10)
    N_list = np.logspace(1, 10, num=10, base=10, dtype=float)
    
    simulated_variances = []
    
    print("Appendix A: 巨大スケールにおけるSignal Cliff計算実験を開始します...")
    print(f"パラメータ: J={J}, H={H}, T={T}, 試行回数={num_trials}\n")

    # --- 2. メインループ: 各システムサイズ N でのシミュレーション ---
    for N in N_list:
        final_magnetizations = np.zeros(num_trials)
        
        for trial in range(num_trials):
            # 初期状態: 平均0、微小な揺らぎ(0.05)を持つ中立状態
            m = np.random.normal(0, 0.05)
            
            for t in range(time_steps):
                # ボルツマン分布に基づく +1(多数派) 選択確率
                p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * (J * m + H)))
                
                # 計算コスト削減のためのマクロ動的近似
                if N < 1e6:
                    # 小・中スケール: 厳密な二項分布サンプリング
                    n_plus = np.random.binomial(n=int(N), p=p_plus)
                else:
                    # 巨大スケール: 中心極限定理に基づく正規分布近似
                    mu = N * p_plus
                    sigma = np.sqrt(N * p_plus * (1.0 - p_plus))
                    n_plus = np.random.normal(loc=mu, scale=sigma)
                    n_plus = np.clip(n_plus, 0, N)
                
                # 磁化 (平均意見) の更新
                m = (2.0 * n_plus - N) / N
                
            final_magnetizations[trial] = m
            
        # 分散の実測値を記録
        var_m = np.var(final_magnetizations)
        simulated_variances.append(var_m)
        
        print(f"N = 10^{int(np.log10(N)):2d} | 最終分散: {var_m:.4e}")

    # --- 3. 理論的スケーリング則 (1/N) の計算 ---
    # 収束後の定常状態における確率 p_steady を推定（最大のNの結果を使用）
    m_steady = np.mean(final_magnetizations)
    p_steady = 1.0 / (1.0 + np.exp(-2.0 * beta * (J * m_steady + H)))
    
    # 理論分散: σ_m^2 = 4 * p * (1 - p) / N
    theoretical_variances = (4.0 * p_steady * (1.0 - p_steady)) / N_list

    print("\nシミュレーション完了。グラフを描画します...")

    # --- 4. 結果の可視化 (Log-Log プロット) ---
    plt.figure(figsize=(10, 7))
    
    # 実測値のプロット
    plt.loglog(N_list, simulated_variances, marker='o', markersize=8, color='purple', 
               linewidth=2, label='Simulated Variance (Empirical)')
    
    # 理論値のプロット (Appendix Aの数式証明)
    plt.loglog(N_list, theoretical_variances, linestyle='--', color='black', 
               linewidth=2, label=r'Theoretical Scaling: $\sigma_m^2 \propto 1/N$')
    
    # 臨界閾値 Nc の可視化
    plt.axvline(x=1e4, color='red', linestyle=':', linewidth=2, alpha=0.8, 
                label='Signal Cliff ($N_c = 10^4$)')
    
    # グラフの装飾
    plt.title('Appendix A: Validation of Signal Cliff and Variance Collapse', fontsize=16)
    plt.xlabel('System Size $N$ (Log Scale)', fontsize=14)
    plt.ylabel('Variance of Final Opinion $\sigma_m^2$ (Log Scale)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    
    # 背景の塗りつぶし（フェーズの視覚化）
    plt.axvspan(10, 1e4, color='green', alpha=0.05, label='Stochastic Phase (Diversity)')
    plt.axvspan(1e4, 1e10, color='red', alpha=0.05, label='Deterministic Phase (Tyranny)')
    
    plt.tight_layout()
    plt.show()

# コードの実行
simulate_appendix_a_signal_cliff()
