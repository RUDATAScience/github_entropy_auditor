import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import warnings

# 警告の非表示（正規近似時の微小な計算誤差ワーニング等を防ぐため）
warnings.filterwarnings('ignore')

def simulate_galam_model():
    """
    Galamモデルに基づくオピニオンダイナミクス・シミュレーション
    Signal Cliffとシャノンエントロピーの崩壊を検証する
    """
    # --- 1. パラメータ設定 ---
    J = 1.0          # 同調圧力 (Conformity pressure)
    H = 0.02         # 外部バイアス (Systemic bias): わずかに多数派へ誘導する微小な力
    T = 1.2          # 社会的温度 (Social temperature): ノイズや個人の気まぐれ
    beta = 1.0 / T
    
    time_steps = 50  # 収束までの時間ステップ
    num_trials = 500 # 各システムサイズ N における試行回数（独立した社会の数）

    # 検証するシステムサイズ N のリスト (10^1 から 10^10)
    N_list = np.logspace(1, 10, num=10, base=10, dtype=float)
    
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
                # 多数派 (+1) を選択する確率のボルツマン分布
                p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * (J * m + H)))
                
                # 計算コスト削減のための動的近似アルゴリズム
                if N < 1e6:
                    # 小・中スケール: 厳密な二項分布サンプリング
                    n_plus = np.random.binomial(n=int(N), p=p_plus)
                else:
                    # 巨大スケール: 中心極限定理に基づく正規分布近似
                    mu = N * p_plus
                    sigma = np.sqrt(N * p_plus * (1.0 - p_plus))
                    n_plus = np.random.normal(loc=mu, scale=sigma)
                    n_plus = np.clip(n_plus, 0, N) # Nの範囲内に収める
                
                # 磁化 (平均意見) の更新: m ∈ [-1, 1]
                m = (2.0 * n_plus - N) / N
                
            final_magnetizations[trial] = m
            
        # 分散の計算
        var_m = np.var(final_magnetizations)
        variances.append(var_m)
        
        # シャノンエントロピーの計算
        # 最終的な意見分布のヒストグラムから確率密度を推定
        hist, bin_edges = np.histogram(final_magnetizations, bins=50, density=True)
        # 確率が0のビンを除外してエントロピーを計算
        hist_prob = hist[hist > 0] * np.diff(bin_edges)[hist > 0]
        S = entropy(hist_prob, base=2) # 単位はビット (bits)
        entropies.append(S)
        
        print(f"N = 10^{int(np.log10(N))}: 分散 = {var_m:.2e}, エントロピー = {S:.2f} bits")

    print("シミュレーション完了。グラフを描画します。")

    # --- 3. 結果の可視化 (プロット) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # グラフ1: 分散の崩壊 (Signal Cliff)
    ax1.plot(N_list, variances, marker='o', color='purple', linestyle='-')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.axvline(x=1e4, color='red', linestyle='--', alpha=0.7, label='Signal Cliff ($N=10^4$)')
    ax1.set_title('Variance Collapse: The "Signal Cliff"')
    ax1.set_xlabel('Population Size ($N$)')
    ax1.set_ylabel('Variance of Final Opinion (log scale)')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # グラフ2: シャノンエントロピーの崩壊
    ax2.plot(N_list, entropies, marker='^', color='green', linestyle='-')
    ax2.set_xscale('log')
    ax2.axvline(x=1e4, color='red', linestyle='--', alpha=0.7, label='Signal Cliff ($N=10^4$)')
    ax2.set_title('Shannon Entropy Collapse\n(Proof of Informational Death)')
    ax2.set_xlabel('Population Size ($N$)')
    ax2.set_ylabel('Information Entropy (bits)')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# 関数の実行
simulate_galam_model()
