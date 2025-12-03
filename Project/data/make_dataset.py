import pandas as pd
import os
import re
from tqdm import tqdm

def split_sentences_regex(text):
    """
    [문장 분리 로직]
    줄바꿈이 없는 텍스트에서 마침표(.), 물음표(?), 느낌표(!) 뒤에 
    공백이 있을 경우 강제로 줄바꿈을 넣어 분리합니다.
    """
    text = str(text)
    # [.?!] 뒤에 공백(\s+)이 나오면, 구두점 뒤에 줄바꿈(\n)을 삽입
    text = re.sub(r'([.?!])\s+', r'\1\n', text)
    sentences = text.split('\n')
    return [s.strip() for s in sentences if s.strip()]

def process_and_window(file_path, target_label, text_col, split_mode='newline'):
    """
    CSV 파일을 읽어 3-Window 데이터로 변환
    """
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()

    print(f"📂 로드 중: {os.path.basename(file_path)} (Target: {target_label})")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"   Note: UTF-8 실패, CP949로 재시도합니다.")
        df = pd.read_csv(file_path, encoding='cp949')

    if text_col not in df.columns:
        print(f"⚠️ '{text_col}' 컬럼이 없습니다. (존재하는 컬럼: {list(df.columns)})")
        return pd.DataFrame()

    window_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Label {target_label} 처리"):
        # ID 생성
        file_id = row['id'] if 'id' in df.columns else f"doc_{target_label}_{idx}"
        raw_text = row[text_col]

        # 분리 모드 적용
        if split_mode == 'newline':
            sentences = str(raw_text).split('\n')
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            sentences = split_sentences_regex(raw_text)

        if not sentences:
            continue

        # Sliding Window 적용
        for i in range(len(sentences)):
            curr = sentences[i]
            prev1 = sentences[i-1] if i - 1 >= 0 else None
            prev2 = sentences[i-2] if i - 2 >= 0 else None

            window_data.append({
                'id': file_id,
                'prev2': prev2,
                'prev1': prev1,
                'curr': curr,
                'label_class': target_label
            })
    
    return pd.DataFrame(window_data)

def main():
    # ----------------------------------------------------------------
    # 1. 경로 설정 (data 폴더 내부 기준)
    # ----------------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    phishing_path = os.path.join(current_dir, "phishing_data.csv")
    normal_path = os.path.join(current_dir, "KorCCVi_v2.csv")
    output_path = os.path.join(current_dir, "dataset.csv")

    # ----------------------------------------------------------------
    # 2. 데이터 변환 (모든 데이터를 우선 윈도우로 만듦)
    # ----------------------------------------------------------------
    
    # (A) 피싱 데이터 (Label 1)
    df_phishing = process_and_window(
        phishing_path, 
        target_label=1, 
        text_col='script', 
        split_mode='newline'
    )

    # (B) 일반 데이터 (Label 0)
    df_normal = process_and_window(
        normal_path, 
        target_label=0, 
        text_col='transcript', 
        split_mode='regex'
    )

    count_phishing = len(df_phishing)
    count_normal = len(df_normal)

    print(f"\n📊 윈도우 생성 결과: Phishing={count_phishing:,}, Normal={count_normal:,}")

    if count_phishing == 0 or count_normal == 0:
        print("❌ 어느 한쪽의 데이터가 없습니다. CSV 파일 내용을 확인해주세요.")
        return

    # ----------------------------------------------------------------
    # 3. 데이터 밸런싱 (Label 0 무작위 선택)
    # ----------------------------------------------------------------
    # 두 데이터 중 적은 쪽의 개수를 구합니다.
    target_count = min(count_phishing, count_normal)
    
    print(f"\n⚖️ 밸런싱 수행: 각 클래스 당 {target_count:,}개를 맞춥니다.")
    
    # [Label 1 처리]
    # 피싱 데이터가 더 많다면 랜덤하게 줄이고, 적다면(target_count와 같다면) 그대로 씁니다.
    df_phishing_bal = df_phishing.sample(n=target_count, random_state=42)

    # [Label 0 처리 - 요청사항]
    # Label 0 데이터 전체 중에서 target_count 개수만큼 '무작위'로 뽑습니다.
    # random_state=42는 랜덤 결과를 고정하여 재현성을 보장합니다. 
    # (매번 완전 다른 랜덤을 원하면 random_state=None 으로 변경하세요)
    print(f"   -> Label 0 데이터 {count_normal:,}개 중 {target_count:,}개를 무작위로 선택합니다.")
    df_normal_bal = df_normal.sample(n=target_count, random_state=42)

    # ----------------------------------------------------------------
    # 4. 병합 및 저장
    # ----------------------------------------------------------------
    df_final = pd.concat([df_phishing_bal, df_normal_bal])
    
    # 전체 데이터를 다시 한 번 무작위로 섞음 (Shuffle)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 최종 컬럼 정리
    final_cols = ['id', 'prev2', 'prev1', 'curr', 'label_class']
    df_final = df_final[final_cols]

    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 최종 데이터셋 생성 완료!")
    print(f"   - 저장 위치: {output_path}")
    print(f"   - 총 데이터 수: {len(df_final):,}")
    print(f"   - Label 구성: 0(Normal)={len(df_normal_bal):,} / 1(Phishing)={len(df_phishing_bal):,}")

if __name__ == "__main__":
    main()