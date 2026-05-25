import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def simulate_softmax_denominator_violence():
    """
    案3: Softmax関数における「規格化定数の暴力（Denominator Underflow）」
    数値安定化テクニック（最大値の減算）が、マイノリティの尤度を
    Float64の限界（e^-745）を超えて完全にゼロに押しつぶす現象を証明する。
    """
    np.random.seed(42)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle('Denominator Violence in Softmax Normalization', fontsize=16)

    csv_file = 'softmax_underflow_results.csv'
    plot_file = 'softmax_underflow_plot.png'
    zip_file = 'softmax_underflow.zip'

    print("Softmax関数のアンダーフローシミュレーションを実行中...")
    
    # マジョリティとマイノリティの尤度（ロジット）の「差（Delta X）」
    # 例：LLMの学習において、頻出単語(マジョリティ)と希少単語(マイノリティ)の重みの差が拡大していく過程
    delta_x_list = np.linspace(0, 800, 400)
    
    results = []

    for delta_x in delta_x_list:
        x_maj = delta_x
        x_min = 0.0  # マイノリティのロジットを基準(0)とする
        
        # 【数値安定化テクニック (Numerical Stability Trick)】
        # 分母の exp(x) が爆発（Overflow）して NaN になるのを防ぐため、
        # 配列の最大値 (ここでは x_maj) を全ての要素から引くのが標準的な実装。
        max_x = max(x_maj, x_min)
        
        shifted_maj = x_maj - max_x  # 常に 0 になる
        shifted_min = x_min - max_x  # -delta_x になる
        
        # 指数関数の適用
        exp_maj = np.exp(shifted_maj)  # 常に 1.0
        exp_min = np.exp(shifted_min)  # Float64の限界を超えると 0.0 になる
        
        # Softmax確率の計算
        denominator = exp_maj + exp_min
        prob_min = exp_min / denominator
        prob_maj = exp_maj / denominator
        
        # グラフ描画用に、完全なゼロになった場合はFloat64の非正規化最小値(1e-323)で代用
        plot_prob_min = prob_min if prob_min > 0.0 else 1e-323
        
        results.append({
            "Logit_Difference_DeltaX": delta_x,
            "Shifted_Minority_Logit": shifted_min,
            "Exp_Minority_Raw": exp_min,
            "Softmax_Prob_Minority": prob_min,
            "Plotting_Prob": plot_prob_min,
            "Is_Absolute_Zero": (prob_min == 0.0)
        })

    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)

    # ==========================================
    # グラフの描画
    # ==========================================
    color_min = '#9b59b6'
    
    # 確率の推移をLogスケールで描画
    ax1.plot(df['Logit_Difference_DeltaX'], df['Plotting_Prob'], color=color_min, lw=3, label='Minority Softmax Probability')
    ax1.set_yscale('log')
    ax1.set_xlabel('Logit Difference ($X_{maj} - X_{min}$)', fontsize=14)
    ax1.set_ylabel('Minority Probability (Log Scale)', color=color_min, fontsize=14)
    ax1.tick_params(axis='y', labelcolor=color_min)
    
    # Float64限界（約745）に警告線を引く
    ax1.axvline(x=745, color='#e74c3c', linestyle='--', lw=2, label='Float64 Underflow Limit ($\Delta X \\approx 745$)')
    
    # 絶対的ゼロ（死）の領域を塗りつぶし
    ax1.axvspan(745, 800, color='#e74c3c', alpha=0.15, label='Absolute Zero Zone (Epistemic Death)')

    ax1.set_title('Collapse of Minority Probability in LLM Softmax Mechanism', fontsize=14)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.show()

    # ==========================================
    # ZIP化とダウンロード
    # ==========================================
    print("\nデータをZIPファイルに圧縮しています...")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_file)
        zipf.write(plot_file)
        
    print(f"圧縮完了: {zip_file}")
    try:
        from google.colab import files
        files.download(zip_file)
        print("ダウンロードを開始しました。")
    except ImportError:
        print(f"ローカル環境のため、カレントディレクトリに '{zip_file}' を保存しました。")

# 実行
simulate_softmax_denominator_violence()
