import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 警告の非表示
warnings.filterwarnings('ignore')

def simulate_centrist_blackhole_and_hysteresis():
    """
    Model C 拡張のハイライト: 
    中道ブラックホール化の相転移、ヒステリシス（履歴効果）、および塑性変形のシミュレーション
    """
    np.random.seed(42)
    
    # --- 1. シミュレーションの基本設定 ---
    N = 2000          # エージェント数（総人口）
    dt = 0.1          # 時間ステップ
    steps = 2500      # 総ステップ数
    time = np.arange(0, steps * dt, dt)
    
    # --- 2. クライシス（危機）のシナリオ設定 ---
    A_t = np.zeros(steps)
    # t=40 から t=100 にかけて危機が最大(1.0)まで増大し、t=100 から t=180 にかけて平時(0.0)へ戻る
    A_t[400:1000] = np.linspace(0.0, 1.0, 600)
    A_t[1000:1800] = np.linspace(1.0, 0.0, 800)
    
    # 記憶（体感危機レベル） A*(t) の初期化
    A_star = np.zeros(steps)
    tau_fast = 5.0    # 危機に対する反応（パニック）は速い
    tau_slow = 40.0   # 危機が去った後の警戒解除（忘却）は遅い
    
    # --- 3. 意見空間と引力パラメータの設定 ---
    P_L, P_C, P_R = -0.8, 0.0, 0.8  # 各極の座標 (左派, 中道, 右派)
    
    # 平時のポテンシャル引力 (W: 強さ, sigma: 寛容度/引力の及ぶ範囲)
    W0_ext = 1.5      # 極端派(左右)の基本引力
    W0_C = 0.3        # 中道の基本引力（平時は弱い）
    sigma0_ext = 0.3  # 極端派の基本寛容度
    sigma0_C = 0.4    # 中道の基本寛容度
    
    # 塑性変形（ベースラインの恒久的な書き換え）のパラメータ
    A_crit = 0.6      # 体感危機 A*(t) がこの値を超えると、トラウマとしてベースラインが変形する
    gamma_plastic = 0.015 # ベースラインが中道へ引っ張られる速度
    
    # エージェントの初期化 (左派と右派に二極化した平和な社会からスタート)
    baseline_L = np.random.normal(P_L, 0.15, N // 2)
    baseline_R = np.random.normal(P_R, 0.15, N // 2)
    B = np.concatenate([baseline_L, baseline_R]) # 個人の固有の帰属(ベースライン)
    I = np.copy(B)                               # 現在の意見
    
    # 記録用配列
    history_I = np.zeros((steps, N))
    center_ratio = np.zeros(steps)
    
    print("シミュレーションを開始します（中道ブラックホールと塑性変形の計算）...")

    # --- 4. メインループ: 共進化ダイナミクスの計算 ---
    for t_idx in range(steps):
        # (A) 体感危機レベル A*(t) の更新 (非対称な緩和時間)
        if t_idx > 0:
            dA = A_t[t_idx] - A_star[t_idx-1]
            tau = tau_fast if dA > 0 else tau_slow
            A_star[t_idx] = A_star[t_idx-1] + (dA / tau) * dt
            
        curr_A_star = A_star[t_idx]
        
        # (B) 引力と寛容度の動的相転移 (クライシスへの応答)
        # 危機が深まると極端派はオワコン化(指数関数的減衰)し、中道が避難所として巨大化する
        W_ext = W0_ext * np.exp(-3.5 * curr_A_star)
        W_C = W0_C + 4.5 * curr_A_star
        
        sig_ext = sigma0_ext * np.exp(-2.0 * curr_A_star)
        sig_C = sigma0_C + 1.5 * np.log1p(curr_A_star)
        
        # (C) 塑性変形 (ヒステリシスによるベースラインの書き換え)
        # 危機が臨界値 A_crit を超えている間、人々の帰属意識 B 自体が中道へ引っ張られる
        if curr_A_star > A_crit:
            B += gamma_plastic * (curr_A_star - A_crit) * (P_C - B) * dt
            
        # (D) 意見 I(t) の更新 (確率微分方程式 Euler-Maruyama法)
        # 各極からのガウス型引力ポテンシャル
        force_L = W_ext * np.exp(-((I - P_L)**2) / (2 * sig_ext**2)) * (P_L - I)
        force_R = W_ext * np.exp(-((I - P_R)**2) / (2 * sig_ext**2)) * (P_R - I)
        force_C = W_C   * np.exp(-((I - P_C)**2) / (2 * sig_C**2))   * (P_C - I)
        
        # ベースライン（自己同一性）への復元力
        force_B = 1.0 * (B - I)
        
        # 個人の気まぐれノイズ
        noise = np.random.normal(0, 0.08, N)
        
        dI = (force_L + force_R + force_C + force_B) * dt + noise * np.sqrt(dt)
        I = np.clip(I + dI, -1.0, 1.0)
        
        # 記録
        history_I[t_idx] = I
        center_ratio[t_idx] = np.mean(np.abs(I) < 0.25) * 100 # |I| < 0.25 を中道層と定義

    print("計算完了。グラフを描画します...")

    # --- 5. 結果の可視化 ---
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Simulation 3: Centrist Blackhole, Hysteresis, and Plastic Deformation', fontsize=18)
    
    # プロット1: クライシスと記憶(体感危機)の推移
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time, A_t, label='Actual Crisis $A(t)$', color='black', lw=2, linestyle='--')
    ax1.plot(time, A_star, label='Perceived Crisis (Memory) $A^*(t)$', color='red', lw=2)
    ax1.fill_between(time, A_crit, A_star, where=(A_star > A_crit), color='red', alpha=0.2, label='Plastic Deformation Zone')
    ax1.set_title('Crisis Level and Asymmetric Memory Effect')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Crisis Magnitude')
    ax1.legend()
    
    # プロット2: 意見の軌跡（社会全体の分極から中道への雪崩）
    ax2 = plt.subplot(2, 2, 2)
    # 描画を軽くするため、100人のエージェントのみプロット
    sample_agents = np.random.choice(N, 100, replace=False)
    ax2.plot(time, history_I[:, sample_agents], color='blue', alpha=0.05)
    ax2.axhline(0, color='black', lw=1, linestyle='--')
    ax2.set_title('Opinion Trajectories of Agents')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Opinion Value (I)')
    ax2.set_ylim(-1.1, 1.1)
    
    # プロット3: ヒステリシス曲線 (A(t) vs 中道層の割合)
    ax3 = plt.subplot(2, 2, 3)
    scatter = ax3.scatter(A_t, center_ratio, c=time, cmap='viridis', s=10, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Time Progression')
    # 経路を示す矢印のアノテーション
    ax3.annotate('Going (Panic)', xy=(0.4, 20), xytext=(0.6, 5), arrowprops=dict(facecolor='black', shrink=0.05))
    ax3.annotate('Returning (Trauma)', xy=(0.4, 80), xytext=(0.2, 90), arrowprops=dict(facecolor='red', shrink=0.05))
    ax3.set_title('Hysteresis Loop of the Centrist Population')
    ax3.set_xlabel('Actual Crisis Level $A(t)$')
    ax3.set_ylabel('Centrist Population Ratio (%)')
    
    # プロット4: 塑性変形の証明（t=0 と t=end の分布比較）
    ax4 = plt.subplot(2, 2, 4)
    sns.kdeplot(history_I[0], color='gray', fill=True, alpha=0.5, label='Initial ($t=0$)', ax=ax4)
    sns.kdeplot(history_I[-1], color='orange', fill=True, alpha=0.5, label='Final (After Crisis)', ax=ax4)
    ax4.set_title('Proof of Plastic Deformation (Irreversible Change)')
    ax4.set_xlabel('Opinion Value (I)')
    ax4.set_ylabel('Probability Density')
    ax4.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 実行
simulate_centrist_blackhole_and_hysteresis()
