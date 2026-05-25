import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def verify_three_thresholds():
    """
    HKモデル、Galamモデル、統合拡張モデルの3つの異なる臨界閾値を計算・比較する
    """
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Critical Thresholds in Three Different Opinion Dynamics Models', fontsize=16)

    # ==========================================
    # 1. Hegselmann-Krause (HK) モデル: 寛容度 ε の閾値検証
    # ==========================================
    print("1/3: HKモデルの寛容度閾値(ε)を計算中...")
    epsilons = np.linspace(0.05, 0.35, 30)
    num_clusters = []
    N_hk = 200
    steps_hk = 50

    for eps in epsilons:
        opinions = np.random.uniform(-1, 1, N_hk)
        for _ in range(steps_hk):
            new_opinions = np.zeros(N_hk)
            for i in range(N_hk):
                # 寛容度 ε 以内の他者の意見の平均をとる
                neighbors = opinions[np.abs(opinions - opinions[i]) <= eps]
                new_opinions[i] = np.mean(neighbors)
            opinions = new_opinions
        
        # 収束後のクラスタ数をカウント（小数点第2位で丸めて一意な数を取得）
        unique_clusters = len(np.unique(np.round(opinions, 2)))
        num_clusters.append(unique_clusters)

    axes[0].plot(epsilons, num_clusters, marker='o', color='blue')
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
    variances = []
    J, H, beta = 1.0, 0.02, 1.0/1.2  # 同調圧力, バイアス, 社会的温度
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
        variances.append(np.var(final_m))

    axes[1].plot(N_list, variances, marker='s', color='purple')
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
    centrist_ratios = []
    N_int = 500
    dt = 0.1
    steps_int = 500

    for A_max in A_max_list:
        # 初期状態: 左右に二極化
        B = np.concatenate([np.random.normal(-0.8, 0.1, N_int//2), 
                            np.random.normal(0.8, 0.1, N_int//2)])
        I = np.copy(B)
        A_crit = 0.6
        gamma_plastic = 0.02
        
        # 危機が一時的に上昇し、その後収束するシナリオ
        for step in range(steps_int):
            curr_A = A_max if (100 < step < 300) else 0.0
            
            # 塑性変形 (A_critを超えた場合のみベースラインが書き換わる)
            if curr_A > A_crit:
                B += gamma_plastic * (curr_A - A_crit) * (0.0 - B) * dt
                
            # 引力とノイズの計算
            W_C = 0.3 + 4.5 * curr_A
            force_C = W_C * np.exp(-(I**2) / (2 * 0.4**2)) * (0.0 - I)
            force_B = 1.0 * (B - I)
            dI = (force_C + force_B) * dt + np.random.normal(0, 0.05, N_int) * np.sqrt(dt)
            I = np.clip(I + dI, -1.0, 1.0)
            
        # 最終的な中道層(|I| < 0.25)の割合
        centrist_ratio = np.mean(np.abs(I) < 0.25) * 100
        centrist_ratios.append(centrist_ratio)

    axes[2].plot(A_max_list, centrist_ratios, marker='^', color='orange')
    axes[2].axvline(x=0.6, color='red', linestyle='--', label=r'Plastic Threshold $A_{crit} = 0.6$')
    axes[2].set_title('3. Integrated Model: Crisis Threshold ($A$)')
    axes[2].set_xlabel('Maximum Crisis Level $A_{max}$')
    axes[2].set_ylabel('Final Centrist Ratio (%)')
    axes[2].grid(True, ls="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("計算完了。")

# 実行
verify_three_thresholds()
