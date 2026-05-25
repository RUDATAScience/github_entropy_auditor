import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
import warnings

warnings.filterwarnings('ignore')

def simulate_curse_of_dimensionality():
    """
    案2: 高次元ベクトル空間(Embeddings)における「距離の崩壊」シミュレーション
    次元数Dの増大に伴い、マジョリティ内の距離分散が消失し、
    明確なシグナルを持つマイノリティとの距離差がノイズに埋没する現象を証明する。
    """
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Curse of Dimensionality in Vector Embeddings', fontsize=16)

    csv_file = 'embedding_curse_results.csv'
    plot_file = 'embedding_curse_plot.png'
    zip_file = 'embedding_curse.zip'

    print("高次元空間における距離の崩壊をシミュレーション中...")
    
    # テストする次元数 D (2次元の平面から、LLMクラスの2000次元まで)
    D_list = np.logspace(0.301, 3.301, num=20, base=10, dtype=int)
    
    N_maj = 500  # マジョリティのサンプル数
    N_min = 50   # マイノリティのサンプル数
    
    # マイノリティが持つ「明確な意味的シグナル」の強度（特定次元へのシフト量 S）
    # 例：特定のキーワードや主張を持っていることを示す距離
    signal_shift_S = 6.0 
    
    results = []

    for D in D_list:
        # 1. マジョリティ・クラスターの生成（D次元標準正規分布）
        X_maj = np.random.normal(0, 1, (N_maj, D))
        
        # マジョリティ内のペアワイズ距離（ユークリッド距離）
        dists_maj = pdist(X_maj, metric='euclidean')
        mean_maj = np.mean(dists_maj)
        std_maj = np.std(dists_maj)
        
        # 2. マイノリティ・クラスターの生成
        # マジョリティから「シグナル S」だけ離れた位置に生成
        shift_vector = np.zeros(D)
        shift_vector[0] = signal_shift_S  # 1つの次元に強烈な特徴があると仮定
        X_min = np.random.normal(0, 1, (N_min, D)) + shift_vector
        
        # マイノリティからマジョリティへの距離
        dists_cross = cdist(X_min, X_maj, metric='euclidean').flatten()
        mean_cross = np.mean(dists_cross)
        
        # 3. 識別可能性（Zスコア）の計算
        # マジョリティ内のノイズ（標準偏差）に対して、マイノリティが何シグマ離れているか
        z_score = (mean_cross - mean_maj) / std_maj
        
        # 相対分散（変動係数 CV = Std / Mean）
        relative_variance = std_maj / mean_maj
        
        results.append({
            "Dimension_D": D,
            "Intra_Majority_Mean": mean_maj,
            "Intra_Majority_Std": std_maj,
            "Relative_Variance_CV": relative_variance,
            "Cross_Minority_Mean": mean_cross,
            "Separation_Z_Score": z_score
        })

    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)

    # ==========================================
    # 左図: 相対分散（CV）の崩壊
    # ==========================================
    axes[0].plot(df['Dimension_D'], df['Relative_Variance_CV'], marker='o', color='#8e44ad', lw=2)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title('1. Collapse of Relative Variance (Std / Mean)')
    axes[0].set_xlabel('Dimensionality $D$ (Log Scale)')
    axes[0].set_ylabel('Relative Variance of Distances')
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    # ==========================================
    # 右図: 識別可能性（Zスコア）の崩壊
    # ==========================================
    axes[1].plot(df['Dimension_D'], df['Separation_Z_Score'], marker='s', color='#c0392b', lw=2)
    # Z=3 (統計的に識別可能な限界) に線を引く
    axes[1].axhline(y=3.0, color='black', linestyle='--', label='Detection Threshold ($Z = 3$)')
    axes[1].set_xscale('log')
    axes[1].set_title('2. Collapse of Minority Separation (Z-Score)')
    axes[1].set_xlabel('Dimensionality $D$ (Log Scale)')
    axes[1].set_ylabel('Separation Distance in Sigmas (Z-Score)')
    axes[1].fill_between(df['Dimension_D'], 0, 3, color='red', alpha=0.1, label='Indistinguishable Zone')
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
simulate_curse_of_dimensionality()
