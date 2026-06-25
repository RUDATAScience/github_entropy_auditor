import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
import scipy.stats as stats
try:
    from google.colab import files
except ImportError:
    pass # ローカル環境実行用

# 出力用ディレクトリの作成
OUTPUT_DIR = "simulation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# グラフの全体設定 (論文用)
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'font.family': 'serif'
})

print("Starting simulations...")

# =====================================================================
# Experiment 1: Geometry (Topological Collapse)
# 次元数の増加に伴う距離の分散の縮小（次元の呪い）を検証
# =====================================================================
def run_geometry_simulation():
    dimensions = [64, 256, 1536] # 1536はOpenAI等の標準的な埋め込み次元数
    n_samples = 10000
    sigma_sq = 1.0
    
    plt.figure(figsize=(8, 6))
    results = {}
    
    for D in dimensions:
        # マジョリティの中心(原点)からマイノリティベクトルまでの距離を計算
        # 各次元は N(0, sigma_sq) に従うと仮定（距離の二乗はカイ二乗分布的に振る舞う）
        minority_vectors = np.random.normal(loc=0.0, scale=np.sqrt(sigma_sq), size=(n_samples, D))
        # 正規化された二乗距離: ||v_m - mu_d||^2 / D
        distances_sq_norm = np.sum(minority_vectors**2, axis=1) / D
        
        # 確率密度関数をプロット
        sns_kde_x = np.linspace(0, 3, 500)
        kde = stats.gaussian_kde(distances_sq_norm)
        plt.plot(sns_kde_x, kde(sns_kde_x), label=f'D = {D}', lw=2)
        
        # データを保存
        results[f'D_{D}'] = distances_sq_norm
        
    plt.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Expected Value')
    plt.title("Exp 1: Topological Collapse (Concentration of Distance)")
    plt.xlabel(r"Normalized Squared Distance $||v_m - \mu_d||^2 / D$")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{OUTPUT_DIR}/1_geometry_collapse.png")
    pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/1_geometry_stats.csv", index=False)
    plt.close()
    print("Experiment 1 completed.")

# =====================================================================
# Experiment 2 & 3: EVT (Logit Expansion), Hardware Limits & Solution
# サンプルサイズの増加に伴うロジット差の拡大とFloat64アンダーフローを検証
# =====================================================================
def run_evt_hardware_simulation():
    # パラメータ設定
    beta = 36.0 # スケールパラメータ (Gumbel分布)
    N_minority = 10**3
    
    # マジョリティのサンプルサイズ N_d を 10^4 から 10^12 まで対数的に変化
    N_majority_range = np.logspace(4, 12, num=100)
    
    # 式(14): Delta z = beta * ln(N_d / N_m)
    delta_z = beta * np.log(N_majority_range / N_minority)
    
    # IEEE 754 Float64 における確率ウェイト P = exp(-Delta z)
    # np.exp は -745 以下で厳密に 0.0 を返す（Subnormal underflow）
    probability_weight = np.exp(-delta_z)
    
    # データをDataFrameにまとめる
    df_evt = pd.DataFrame({
        'N_majority': N_majority_range,
        'Log10_N_majority': np.log10(N_majority_range),
        'Delta_Z': delta_z,
        'Probability_Weight': probability_weight
    })
    
    # ------ Plot 2: EVT Expansion and Float64 Limit ------
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    ax1.plot(np.log10(N_majority_range), delta_z, 'b-', lw=2.5, label=r'Logit Gap $\Delta z$')
    ax1.set_xlabel(r"Global Majority Sample Size ($\log_{10} N_d$)")
    ax1.set_ylabel(r"Logit Gap $\Delta z$", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # ハードウェア限界線 (Float64 Underflow)
    ax1.axhline(745.13, color='r', linestyle='--', lw=2, label='Float64 Underflow Limit (745.13)')
    
    # マイクロチャンキングの制約領域
    ax1.axvspan(3, 4, color='green', alpha=0.2, label='Micro-chunking Range ($N_{chunk} \leq 10^4$)')
    
    # 第2軸: 生存確率 (Softmax Freezing)
    ax2 = ax1.twinx()
    ax2.plot(np.log10(N_majority_range), probability_weight, 'k-.', lw=2, label=r'Minority Prob. Weight $P(z_m)$')
    ax2.set_ylabel(r"Probability Weight $\exp(-\Delta z)$", color='k')
    ax2.set_yscale('log')
    ax2.set_ylim([1e-330, 1])
    
    fig.suptitle("Exp 2 & 3: Softmax Freezing and Micro-chunking Effect")
    
    # 凡例を統合
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{OUTPUT_DIR}/2_evt_hardware_solution.png")
    df_evt.to_csv(f"{OUTPUT_DIR}/2_evt_hardware_stats.csv", index=False)
    plt.close()
    print("Experiment 2 & 3 completed.")

# 実行
run_geometry_simulation()
run_evt_hardware_simulation()

# =====================================================================
# ZIPファイルへの圧縮とダウンロード
# =====================================================================
ZIP_NAME = "simulation_results_informational_health"
shutil.make_archive(ZIP_NAME, 'zip', OUTPUT_DIR)
print(f"\nAll simulations completed. Results saved and compressed to {ZIP_NAME}.zip")

# Colab環境であれば自動ダウンロードをトリガー
try:
    files.download(f"{ZIP_NAME}.zip")
except Exception as e:
    print("Please download the zip file manually from the Colab file browser.")
