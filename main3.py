import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# 警告の非表示
warnings.filterwarnings('ignore')

def simulate_crisis_with_statistics():
    """
    複数試行によるモンテカルロ・シミュレーションと統計学的有意性の検証
    結果をCSVに出力し、分布の可視化を行う
    """
    # --- 1. シミュレーションの基本設定 ---
    dt = 0.01          # 時間ステップ
    steps = 4000       # 総ステップ数
    time = np.linspace(0, 40, steps)
    num_trials = 100   # 各シナリオの試行回数 (統計的有意性を出すためのN数)
    
    # 共通パラメータ
    epsilon = 0.5      # 寛容度の閾値
    gamma_inf = 3.0    # 雪崩の基本速度
    alpha = 0.05       # 最初の火種係数
    beta_0 = 0.2       # 平時の同調圧力
    kappa = 8.0        # クライシス時のパニック係数
    
    # 記録用データ構造
    results_data = []
    # 描画用に最初の数回分の軌跡だけ保存する辞書
    trajectories = {"Normal": [], "Crisis": []}
    
    print(f"シミュレーションを開始します（各シナリオ {num_trials} 回の試行）...")

    # --- 2. メインループ: 複数試行の実行 ---
    for scenario in ["Normal", "Crisis"]:
        for trial in range(num_trials):
            I_G = np.zeros(steps)
            I_I = np.zeros(steps)
            x = np.zeros(steps)
            D_GI = np.zeros(steps)
            
            # 初期状態: ゆらぎ（ノイズ）を持たせる
            I_G[0] = np.random.normal(0.2, 0.05)  # 平均0.2, 標準偏差0.05の中道
            I_I[0] = 0.8
            x[0] = 0.99
            
            for i in range(steps - 1):
                t = time[i]
                
                # クライシスレベル
                A = 1.0 if (scenario == "Crisis" and t >= 10.0) else 0.0
                    
                ig = I_G[i]
                ii = I_I[i]
                xi = x[i]
                
                # --- 動的ネットワーク方程式 ---
                delta = abs(ig - ii)
                trigger = 1.0 if delta > epsilon else 0.0
                beta = beta_0 + kappa * A
                
                # 相互作用の微小なノイズ
                noise_x = np.random.normal(0, 0.002)
                dxdt = -gamma_inf * xi * (alpha * trigger + beta * (1.0 - xi))
                x_next = max(0.0, min(1.0, xi + dxdt * dt + noise_x))
                
                # 信頼係数
                d_gi = 2.0 * x_next - 1.0
                
                # --- 意見ダイナミクス方程式 ---
                noise_ig = np.random.normal(0, 0.005) # 意見の微小な揺らぎ
                dIg_dt = d_gi * (ii - ig) * 0.8 - 0.1 * ig
                
                I_G[i+1] = ig + dIg_dt * dt + noise_ig
                I_I[i+1] = ii
                x[i+1] = x_next
                D_GI[i+1] = d_gi
                
            # プロット用に各シナリオ10回分だけ軌跡を保存
            if trial < 10:
                trajectories[scenario].append(I_G)
                
            # 統計解析用に最終状態を記録
            results_data.append({
                "Scenario": scenario,
                "Trial_ID": trial + 1,
                "Final_Opinion_IG": I_G[-1],
                "Final_Trust_DGI": D_GI[-1],
                "Follow_Ratio_x": x[-1]
            })

    # --- 3. データフレーム化とCSV保存 ---
    df = pd.DataFrame(results_data)
    csv_filename = "simulation_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n全試行のデータを '{csv_filename}' に保存しました。")

    # --- 4. 統計学的有意性の検定 (t検定) ---
    normal_ig = df[df["Scenario"] == "Normal"]["Final_Opinion_IG"]
    crisis_ig = df[df["Scenario"] == "Crisis"]["Final_Opinion_IG"]
    
    # Welchのt検定 (分散が等しいと仮定しない)
    t_stat, p_val = stats.ttest_ind(normal_ig, crisis_ig, equal_var=False)
    
    print("\n--- 統計解析結果 ---")
    print(f"Normal時 最終意見(I_G) - 平均: {normal_ig.mean():.3f}, 標準偏差: {normal_ig.std():.3f}")
    print(f"Crisis時 最終意見(I_G) - 平均: {crisis_ig.mean():.3f}, 標準偏差: {crisis_ig.std():.3f}")
    print(f"Welchのt検定 p値: {p_val:.2e}")
    if p_val < 0.01:
        print("=> p < 0.01 です。「クライシスの有無による社会的分断の差」は統計学的に極めて有意です。")
    else:
        print("=> 有意な差は確認されませんでした。")

    # --- 5. 結果の可視化 ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左図: 意見の軌跡サンプル (各シナリオ10回分)
    for idx, traj in enumerate(trajectories["Normal"]):
        lbl = "Normal (Public)" if idx == 0 else None
        axes[0].plot(time, traj, color='green', alpha=0.3, label=lbl)
    for idx, traj in enumerate(trajectories["Crisis"]):
        lbl = "Crisis (Public)" if idx == 0 else None
        axes[0].plot(time, traj, color='red', alpha=0.3, label=lbl)
        
    axes[0].axvline(10, color='black', linestyle=':', label='Crisis Trigger (t=10)')
    axes[0].set_title('Sample Trajectories of Public Opinion $I_G$ (10 trials)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Opinion Value')
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].legend()
    
    # 右図: 最終意見の分布と統計的差異 (Boxplot)
    sns.boxplot(x="Scenario", y="Final_Opinion_IG", data=df, ax=axes[1], palette=["#2ecc71", "#e74c3c"])
    sns.stripplot(x="Scenario", y="Final_Opinion_IG", data=df, ax=axes[1], color='black', alpha=0.3, jitter=True)
    
    # p値の注釈を追加
    axes[1].set_title(f'Distribution of Final Opinion\n(p-value: {p_val:.2e})')
    axes[1].set_ylabel('Final Public Opinion $I_G$')
    
    plt.tight_layout()
    plt.show()

# 実行
simulate_crisis_with_statistics()
