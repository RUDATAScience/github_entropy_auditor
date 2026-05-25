import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def simulate_modularity_resolution_limit():
    """
    案1: ネットワーク・クラスタリングにおける「モジュラリティの解像度限界」
    ネットワーク全体の規模 N が増大するにつれ、小さなコミュニティが
    アルゴリズムによって強制的に併合（不可視化）される現象をシミュレートする。
    """
    np.random.seed(42)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    csv_file = 'modularity_resolution_results.csv'
    plot_file = 'network_resolution_plot.png'
    zip_file = 'network_resolution_limit.zip'

    # ==========================================
    # シミュレーション設定
    # ==========================================
    print("ネットワーククラスタリングの解像度限界を計算中...")
    
    # ネットワークのノード数 N を 1,000 から 10,000,000 まで対数的に増加
    N_list = np.logspace(3, 7, num=50, base=10, dtype=int)
    average_degree = 10 # 1人あたりの平均フォロー/リポスト数
    
    # 観察対象とするマイノリティ・コミュニティの内部エッジ数 (規模)
    minority_community_sizes = [50, 200, 500, 2000, 5000]
    
    results = []

    for N in N_list:
        # ネットワーク全体の総エッジ数 L
        L = (N * average_degree) / 2.0
        
        # モジュラリティ最適化（Louvain法など）が識別できる最小コミュニティサイズの理論限界
        # Fortunato & Barthelemy (2007) より: l_c \approx \sqrt{L/2}
        min_detectable_edges = np.sqrt(L / 2.0)
        
        # 各マイノリティ・コミュニティが生存（独立したコミュニティとして検出）できるか判定
        survival_status = {}
        surviving_count = 0
        
        for size in minority_community_sizes:
            # コミュニティの内部エッジ数が解像度限界を上回っていれば生存
            is_surviving = size > min_detectable_edges
            survival_status[f"Size_{size}_Survives"] = is_surviving
            if is_surviving:
                surviving_count += 1
                
        # 結果の記録
        row = {
            "Total_Nodes_N": N,
            "Total_Edges_L": L,
            "Min_Detectable_Edges": min_detectable_edges,
            "Surviving_Minority_Communities": surviving_count
        }
        row.update(survival_status)
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)

    # ==========================================
    # グラフの描画
    # ==========================================
    color_limit = '#c0392b'
    ax1.plot(df['Total_Nodes_N'], df['Min_Detectable_Edges'], color=color_limit, lw=3, label='Resolution Limit (Min Detectable Size)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Total Network Nodes $N$ (Log Scale)', fontsize=14)
    ax1.set_ylabel('Minimum Detectable Community Size (Edges)', color=color_limit, fontsize=14)
    ax1.tick_params(axis='y', labelcolor=color_limit)
    
    # 各マイノリティコミュニティのサイズを水平線で描画
    colors = ['#bdc3c7', '#95a5a6', '#7f8c8d', '#34495e', '#2c3e50']
    for i, size in enumerate(minority_community_sizes):
        ax1.axhline(y=size, color=colors[i], linestyle='--', label=f'Minority Community (Size={size})')
        
    ax1.set_title('Resolution Limit of Modularity in Network Clustering', fontsize=16)
    
    # 生存コミュニティ数を第2軸に描画
    ax2 = ax1.twinx()
    color_survive = '#2980b9'
    ax2.plot(df['Total_Nodes_N'], df['Surviving_Minority_Communities'], color=color_survive, lw=2, linestyle='-.', label='Surviving Communities')
    ax2.set_ylabel('Number of Surviving Minority Communities', color=color_survive, fontsize=14)
    ax2.tick_params(axis='y', labelcolor=color_survive)
    ax2.set_ylim(-0.5, 5.5)

    # 凡例の統合
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
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
simulate_modularity_resolution_limit()
