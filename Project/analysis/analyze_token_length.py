import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer

# -------------------------------------------------------------------------
# 1. 경로 설정
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.utils.text_utils import make_window_text

# -------------------------------------------------------------------------
# 2. 설정
# -------------------------------------------------------------------------
DATA_PATH = "../data/dataset.csv"
MODEL_NAME = "klue/roberta-base"
CURRENT_MAX_LEN = 128

def analyze_lengths():
    if not os.path.exists(os.path.join(current_dir, DATA_PATH)):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        return

    print(f"📂 데이터 로드 중: {DATA_PATH}")
    try:
        df = pd.read_csv(os.path.join(current_dir, DATA_PATH))
    except Exception as e:
        print(f"CSV 읽기 오류: {e}")
        return

    print(f"🤖 토크나이저 로드: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[START]']})
    
    token_counts = []
    long_samples = [] 
    
    print("📊 텍스트 길이 분석 중...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        full_text = make_window_text(
            prev2=row.get('prev2'),
            prev1=row.get('prev1'),
            curr=row['curr'],
            sep_token=tokenizer.sep_token,
            start_marker="[START]"
        )
        
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        length = len(tokens)
        token_counts.append(length)

        # [수정됨] Label 정보도 같이 저장
        if length > CURRENT_MAX_LEN:
            long_samples.append({
                'id': row['id'],
                'length': length,
                'label': row['label_class'], # 라벨 추가 (0 or 1)
                'prev2': row.get('prev2'),
                'prev1': row.get('prev1'),
                'curr': row['curr']
            })

    token_counts = np.array(token_counts)

    # -------------------------------------------------------------------------
    # 3. 초과 데이터 리포트
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print(f"🚨 길이 {CURRENT_MAX_LEN} 초과 데이터 분석")
    print("="*50)
    
    if len(long_samples) > 0:
        long_df = pd.DataFrame(long_samples)
        long_df = long_df.sort_values(by='length', ascending=False)
        
        # 라벨별 분포 확인
        label_counts = long_df['label'].value_counts()
        print(f"총 {len(long_df):,}개의 데이터가 기준을 초과했습니다.")
        print(f" - Phishing(1): {label_counts.get(1, 0):,} 개")
        print(f" - Normal(0)  : {label_counts.get(0, 0):,} 개")
        
        print("\n[가장 긴 데이터 TOP 3 예시]")
        for i, row in long_df.head(3).iterrows():
            label_str = "피싱(1)" if row['label'] == 1 else "일반(0)"
            print(f"🔴 ID: {row['id']} [{label_str}] (Length: {row['length']})")
            print(f"   - prev2: {str(row['prev2'])[:50]}...") 
            print(f"   - prev1: {str(row['prev1'])[:50]}...")
            print(f"   - curr : {str(row['curr'])[:50]}...")
            print("-" * 30)

        # CSV 파일 저장
        save_csv_path = os.path.join(current_dir, "long_samples_report.csv")
        
        # [수정됨] 컬럼 순서에 label 추가
        cols = ['id', 'length', 'label', 'prev2', 'prev1', 'curr']
        long_df = long_df[cols]
        
        long_df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 상세 리포트가 저장되었습니다: {save_csv_path}")
        
    else:
        print(f"🎉 {CURRENT_MAX_LEN}을 초과하는 데이터가 없습니다.")

    # -------------------------------------------------------------------------
    # 4. 시각화 (동일)
    # -------------------------------------------------------------------------
    if os.name == 'posix': plt.rc('font', family='AppleGothic')
    else: plt.rc('font', family='Malgun Gothic')
    plt.rc('axes', unicode_minus=False)

    plt.figure(figsize=(12, 6))
    sns.histplot(token_counts, bins=50, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(x=CURRENT_MAX_LEN, color='red', linestyle='--', linewidth=2, label=f'Current Max ({CURRENT_MAX_LEN})')
    plt.axvline(x=np.percentile(token_counts, 99), color='green', linestyle=':', linewidth=2, label='99% Cut-off')
    
    plt.title(f'Token Length Distribution ({MODEL_NAME})')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    analyze_lengths()