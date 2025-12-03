import sys
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer

# -------------------------------------------------------------------------
# 1. 설정 (Configuration)
# -------------------------------------------------------------------------
# 분석할 텍스트 파일들이 있는 폴더 (data/text_data)
# analysis 폴더에서 실행한다고 가정하고 상위 폴더(..)로 나갑니다.
TEXT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/text_data")
REPORT_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_sentences_report.csv")
IMG_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_data_histogram.png")

MODEL_NAME = "klue/roberta-base"
THRESHOLD_LEN = 128

# -------------------------------------------------------------------------
# 2. 유틸리티 함수
# -------------------------------------------------------------------------
def split_sentences(text):
    """
    텍스트를 문장 단위로 분리합니다.
    1. 줄바꿈(\n)이 있으면 우선적으로 나눕니다.
    2. 문장 부호(. ? !) 뒤에 공백이 있으면 나눕니다.
    """
    text = str(text)
    # 구두점 뒤에 줄바꿈 삽입 (정규식)
    text = re.sub(r'([.?!])\s+', r'\1\n', text)
    # 줄바꿈 기준으로 분리
    sentences = text.split('\n')
    return [s.strip() for s in sentences if s.strip()]

def main():
    # ---------------------------------------------------------------------
    # 3. 준비 단계
    # ---------------------------------------------------------------------
    if not os.path.exists(TEXT_DATA_DIR):
        print(f"❌ 폴더를 찾을 수 없습니다: {TEXT_DATA_DIR}")
        return

    # .txt 파일 목록 가져오기
    file_list = glob.glob(os.path.join(TEXT_DATA_DIR, "*.txt"))
    print(f"📂 '{TEXT_DATA_DIR}' 폴더에서 {len(file_list)}개의 텍스트 파일을 발견했습니다.")

    if len(file_list) == 0:
        print("⚠️ 분석할 텍스트 파일이 없습니다.")
        return

    print(f"🤖 토크나이저 로드: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # ---------------------------------------------------------------------
    # 4. 파일 순회 및 길이 분석
    # ---------------------------------------------------------------------
    all_lengths = []
    long_sentences = [] # 128 토큰 넘는 문장 보관소

    print("📊 파일 분석 중...")
    for file_path in tqdm(file_list):
        file_name = os.path.basename(file_path)
        
        # 파일 읽기 (인코딩 처리)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp949') as f:
                content = f.read()
        
        # 문장 분리
        sentences = split_sentences(content)
        
        # 각 문장별 길이 측정
        for sent in sentences:
            # 토큰화 (길이 측정용)
            tokens = tokenizer.encode(sent, add_special_tokens=True)
            length = len(tokens)
            
            all_lengths.append(length)
            
            # 기준 초과 시 기록
            if length > THRESHOLD_LEN:
                long_sentences.append({
                    'file_name': file_name,
                    'length': length,
                    'sentence': sent  # 문제의 문장 내용
                })

    all_lengths = np.array(all_lengths)

    # ---------------------------------------------------------------------
    # 5. 결과 리포트 (CSV 저장)
    # ---------------------------------------------------------------------
    print("\n" + "="*50)
    print(f"🚨 길이 {THRESHOLD_LEN} 초과 문장 분석 결과")
    print("="*50)

    if len(long_sentences) > 0:
        df_long = pd.DataFrame(long_sentences)
        # 길이 순 정렬
        df_long = df_long.sort_values(by='length', ascending=False)
        
        print(f"총 {len(df_long):,}개의 문장이 {THRESHOLD_LEN} 토큰을 초과했습니다.")
        
        # CSV 저장
        df_long.to_csv(REPORT_SAVE_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ 초과 문장 목록을 저장했습니다: {REPORT_SAVE_PATH}")
        
        print("\n[가장 긴 문장 Top 3]")
        for i, row in df_long.head(3).iterrows():
            print(f"📄 파일: {row['file_name']} (길이: {row['length']})")
            print(f"📝 내용: {row['sentence'][:60]}...\n")
    else:
        print(f"🎉 {THRESHOLD_LEN} 토큰을 초과하는 문장이 하나도 없습니다!")

    # ---------------------------------------------------------------------
    # 6. 히스토그램 시각화
    # ---------------------------------------------------------------------
    print("\n📈 히스토그램 생성 중...")
    
    # 한글 폰트 설정
    if os.name == 'posix': plt.rc('font', family='AppleGothic')
    else: plt.rc('font', family='Malgun Gothic')
    plt.rc('axes', unicode_minus=False)

    plt.figure(figsize=(12, 6))
    sns.histplot(all_lengths, bins=50, kde=True, color='orange', edgecolor='black')
    
    # 기준선
    plt.axvline(x=THRESHOLD_LEN, color='red', linestyle='--', linewidth=2, label=f'Threshold ({THRESHOLD_LEN})')
    if len(all_lengths) > 0:
        p99 = np.percentile(all_lengths, 99)
        plt.axvline(x=p99, color='green', linestyle=':', linewidth=2, label=f'99% Cut-off ({int(p99)})')

    plt.title(f'Sentence Length Distribution (Raw Text Files)')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.savefig(IMG_SAVE_PATH)
    print(f"✅ 히스토그램 이미지를 저장했습니다: {IMG_SAVE_PATH}")
    plt.show()

if __name__ == "__main__":
    main()