import numpy as np
import matplotlib.pyplot as plt
import warnings

# 警告の非表示
warnings.filterwarnings('ignore')

def simulate_crisis_and_social_break():
    """
    Model C 拡張: 政治的クライシスに伴う雪崩現象（フォロー解除カスケード）と
    不信の増幅による社会的分断（Social Break）のシミュレーション
    """
    # --- 1. シミュレーションの基本設定 ---
    dt = 0.01          # 時間ステップ
    steps = 4000       # 総ステップ数
    time = np.linspace(0, 40, steps) # 時間軸 (0から40まで)
    
    # 記録用辞書
    results = {}
    
    # --- 2. 共通パラメータ ---
    epsilon = 0.5      # 寛容度の閾値 (これ以上意見が離れると不満の火種が生まれる)
    gamma_inf = 3.0    # インフルエンサーのフォロー解除の基本速度 (雪崩のスピード)
    alpha = 0.05       # 最初の火種係数 (個人の内面的な気づきによる離脱)
    beta_0 = 0.2       # 平時の同調圧力 (他人が外したから自分も外す傾向)
    kappa = 8.0        # クライシス時の同調圧力増幅係数 (パニック・雪崩のブースト)
    
    print("シミュレーションを開始します（平時 vs クライシス時）...")

    # --- 3. メインループ: 2つのシナリオを計算 ---
    for scenario in ["Normal", "Crisis"]:
        # 状態変数の配列初期化
        I_G = np.zeros(steps)  # 一般大衆の意見 (General Public)
        I_I = np.zeros(steps)  # インフルエンサーの意見 (Influencer)
        x = np.zeros(steps)    # フォロー率 (1.0=全員フォロー, 0.0=全員解除)
        D_GI = np.zeros(steps) # 一般大衆からインフルエンサーへの信頼/不信係数 [-1, 1]
        
        # 初期状態: 一般大衆はやや中道寄り(0.2)、インフルエンサーは極端(0.8)
        # フォロー率はほぼ100%からスタート
        I_G[0] = 0.2
        I_I[0] = 0.8
        x[0] = 0.99
        
        for i in range(steps - 1):
            t = time[i]
            
            # クライシスレベル A(t) の設定
            # シナリオがCrisisの場合、t=10で突然強烈な危機が発生すると仮定
            if scenario == "Crisis" and t >= 10.0:
                A = 1.0  # クライシス発生
            else:
                A = 0.0  # 平時
                
            # 現在の状態
            ig = I_G[i]
            ii = I_I[i]
            xi = x[i]
            
            # --- 動的ネットワーク方程式 (雪崩現象) ---
            # 意見の距離
            delta = abs(ig - ii)
            
            # 発火トリガー: 距離が閾値を超えていれば 1.0 (ヘビサイド関数)
            trigger = 1.0 if delta > epsilon else 0.0
            
            # 動的な同調係数 (危機発生時に爆発的に高まる)
            beta = beta_0 + kappa * A
            
            # フォロー率 x(t) の変化率 (ロジスティック減衰・雪崩方程式)
            dxdt = -gamma_inf * xi * (alpha * trigger + beta * (1.0 - xi))
            
            # xの更新 (0〜1の範囲にクリップ)
            x_next = xi + dxdt * dt
            x_next = max(0.0, min(1.0, x_next))
            
            # --- 信頼/不信係数へのマッピング ---
            # フォロー率 x を 信頼係数 D_GI ∈ [-1, 1] に変換
            # x=1 なら D=1(信頼), x=0 なら D=-1(完全な不信)
            d_gi = 2.0 * x_next - 1.0
            
            # --- 意見ダイナミクス方程式 (Social Break) ---
            # 一般大衆の意見変化:
            # 信頼(D>0)ならインフルエンサーに引かれる。不信(D<0)なら猛反発する。
            # -0.1*ig は中道へ戻ろうとする自然な減衰力
            dIg_dt = d_gi * (ii - ig) * 0.8 - 0.1 * ig
            
            # インフルエンサーは自分の意見を変えない(頑固)と仮定
            dIi_dt = 0.0
            
            # 状態の更新
            I_G[i+1] = ig + dIg_dt * dt
            I_I[i+1] = ii + dIi_dt * dt
            x[i+1] = x_next
            D_GI[i+1] = d_gi
            
        # 結果の保存
        results[scenario] = {
            'I_G': I_G, 'I_I': I_I, 'x': x, 'D_GI': D_GI
        }

    print("シミュレーション完了。グラフを描画します。")

    # --- 4. 結果の可視化 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Political Crisis & Social Break Dynamics (Model C Extended)', fontsize=16)
    
    scenarios = ["Normal", "Crisis"]
    titles = ["Scenario A: Normal Time (No Crisis)", "Scenario B: Crisis Outbreak at t=10"]
    
    for col, scenario in enumerate(scenarios):
        time_ax = time
        res = results[scenario]
        
        # 上段: 雪崩現象 (フォロー率と信頼係数の推移)
        ax1 = axes[0, col]
        ax1.plot(time_ax, res['x'], label='Follow Ratio $x(t)$', color='blue', lw=2)
        ax1.plot(time_ax, res['D_GI'], label='Trust/Distrust $D_{GI}(t)$', color='purple', linestyle='--', lw=2)
        if scenario == "Crisis":
            ax1.axvline(x=10, color='red', linestyle=':', alpha=0.7, label='Crisis Trigger')
        ax1.set_title(titles[col])
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Network State')
        ax1.set_ylim(-1.1, 1.1)
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='lower left')
        
        # 下段: Social Break (意見の分断)
        ax2 = axes[1, col]
        ax2.plot(time_ax, res['I_I'], label='Influencer Opinion $I_I$', color='red', lw=2)
        ax2.plot(time_ax, res['I_G'], label='Public Opinion $I_G$', color='green', lw=2)
        if scenario == "Crisis":
            ax2.axvline(x=10, color='red', linestyle=':', alpha=0.7)
            # 分断の可視化アノテーション
            ax2.annotate('Social Break\n(Repulsion)', xy=(20, res['I_G'][2000]), 
                         xytext=(25, -0.5), arrowprops=dict(facecolor='black', shrink=0.05))
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Opinion Value (I)')
        ax2.set_ylim(-1.0, 1.0)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# シミュレーション実行
simulate_crisis_and_social_break()
