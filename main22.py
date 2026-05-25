import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def kl_divergence(q, p):
    """
    2項分布におけるカルバック・ライブラー情報量 (Kullback-Leibler divergence)
    q: 実際の得票率 (観測された確率)
    p: ランダムな選択における期待値 (1/M)
    """
    q = np.clip(q, 1e-15, 1.0 - 1e-15)
    return q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))

def evaluate_electoral_pressure(N, M, candidate_votes):
    """
    実際の得票数から、その結果をもたらした「圧力の強さ」を統計的に判定する。
    
    N: 有権者数 (または総投票数)
    M: 候補者数
    candidate_votes: 各候補者の名前と得票数の辞書
    """
    p_null = 1.0 / M
    print("=" * 60)
    print(f"【選挙フォレンジック・圧力検定レポート】")
    print(f"有権者総数 (N): {N:,} 人")
    print(f"候補者数   (M): {M} 人 (ランダム期待得票率: {p_null*100:.2f}%)")
    print("-" * 60)
    
    results = []
    
    for name, votes in candidate_votes.items():
        q = votes / N
        
        # 期待値を下回る場合は、マイナス方向の偏りとして扱う
        if q >= p_null:
            d_kl = kl_divergence(q, p_null)
            log10_p = -N * d_kl / np.log(10)
            bias_direction = "上方偏位 (超過)"
        else:
            d_kl = kl_divergence(q, p_null)
            log10_p = -N * d_kl / np.log(10)
            bias_direction = "下方偏位 (過少)"
            
        # 限界閾値に基づく状態判定
        if log10_p > -9:
            status = "正常な揺らぎの範囲"
            pressure_level = "弱 (自然な分散)"
        elif log10_p > -80:
            status = "社会科学的限界を突破"
            pressure_level = "中 (強い同調圧力・強固な組織票)"
        elif log10_p > -308:
            status = "物理的限界を突破"
            pressure_level = "強 (物理的にあり得ない偏り・不正の疑い)"
        else:
            status = "計算機的限界(Underflow)を突破"
            pressure_level = "極 (絶対的統制・データ改ざんの疑い)"
            log10_p = -308.0 # Float64限界クリップ
            
        results.append({
            "候補者": name,
            "得票数": votes,
            "得票率(%)": round(q * 100, 2),
            "偏位の方向": bias_direction,
            "Log10(P)": round(log10_p, 2),
            "圧力レベル": pressure_level,
            "統計的判定": status
        })
        
    df_results = pd.DataFrame(results)
    
    # コンソールへの結果出力
    for idx, row in df_results.iterrows():
        print(f"候補者: {row['候補者']}")
        print(f"  得票数: {row['得票数']:,} 票 ({row['得票率(%)']}%) -> {row['偏位の方向']}")
        print(f"  偶然発生する確率の対数 (Log10 P): {row['Log10(P)']}")
        print(f"  判定: {row['統計的判定']} [{row['圧力レベル']}]")
        print("-" * 60)
        
    return df_results

# ==========================================
# インタラクティブな実行モード
# ==========================================
def run_interactive_forensics():
    print("選挙区のデータを入力してください。")
    try:
        N = int(input("有権者総数（または有効投票総数）を入力: "))
        M = int(input("候補者の数を入力: "))
        
        candidate_votes = {}
        total_votes_entered = 0
        
        for i in range(M):
            name = input(f"\n候補者 {i+1} の名前を入力: ")
            votes = int(input(f"{name} の得票数を入力: "))
            candidate_votes[name] = votes
            total_votes_entered += votes
            
        if total_votes_entered > N:
            print("\n【エラー】得票数の合計が有権者総数を超えています。")
            return
            
        # 検定の実行
        print("\n解析を実行します...\n")
        df = evaluate_electoral_pressure(N, M, candidate_votes)
        
        # CSVへの保存
        csv_filename = "election_pressure_analysis.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\n結果を '{csv_filename}' に保存しました。")
        
    except ValueError:
        print("【エラー】数値を正しく入力してください。")

# スクリプト実行時にインタラクティブモードを起動
if __name__ == "__main__":
    # テスト用のダミーデータを実行したい場合は、以下のコメントアウトを外してください。
    """
    dummy_N = 100000
    dummy_M = 3
    dummy_votes = {
        "候補者A (圧倒的勝利)": 65000,
        "候補者B (通常敗北)": 30000,
        "候補者C (泡沫候補)": 5000
    }
    evaluate_electoral_pressure(dummy_N, dummy_M, dummy_votes)
    """
    
    run_interactive_forensics()
