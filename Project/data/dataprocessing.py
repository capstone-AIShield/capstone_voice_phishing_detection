import os
import glob
import pandas as pd
import uuid

# ==============================================================================
# [설정] 파일 경로 및 컬럼 설정 (사용자 환경에 맞춰 수정하세요)
# ==============================================================================
CONFIG = {
    # 1. 보이스피싱 데이터 (TXT 파일들이 모여있는 폴더 경로)
    'PHISHING_DATA_DIR': './raw_data/phishing_txt', 
    
    # 2. 일반 대화 데이터 (CSV 파일 경로)
    'NORMAL_DATA_PATH': './raw_data/normal_dialogue.csv',
    
    # 3. 일반 대화 CSV에서 '대화 내용'이 담긴 컬럼명 (예: 'text', 'content', 'dialogue' 등)
    'NORMAL_TEXT_COL': 'text', 
    
    # 4. 결과 파일 저장 경로
    'OUTPUT_FILE': './master_dataset.csv'
}

# ==============================================================================
# [함수] 데이터 처리 로직
# ==============================================================================

def load_phishing_data(folder_path):
    """
    폴더 내의 TXT 파일들을 읽어옴 (Label = 1)
    """
    data_list = []
    print(f"[Info] 피싱 데이터 로드 중... ({folder_path})")
    
    # 폴더 내 모든 txt 파일 검색
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if len(file_paths) == 0:
        print(f"   [Warning] 해당 폴더에 .txt 파일이 없습니다.")
        return pd.DataFrame()

    for idx, file_path in enumerate(file_paths):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 통화 내용 전체 읽기 (양쪽 공백 제거)
                content = f.read().strip()
                
                if not content: continue # 빈 파일 건너뜀

                data_list.append({
                    'ID': f"P_{idx+1:05d}",      # 예: P_00001
                    'script': content,           # 통화 내용 전체
                    'label': 1,                  # 피싱 = 1
                    'class': None                # 유형 정보 (보류)
                })
        except Exception as e:
            print(f"   [Error] 파일 읽기 실패 ({file_path}): {e}")
            
    print(f"   -> {len(data_list)}개 파일 로드 완료.")
    return pd.DataFrame(data_list)

def load_normal_data(file_path, text_column):
    """
    CSV 파일을 읽어옴 (Label = 0)
    """
    data_list = []
    print(f"[Info] 일반 데이터 로드 중... ({file_path})")
    
    if not os.path.exists(file_path):
        print(f"   [Error] CSV 파일이 존재하지 않습니다.")
        return pd.DataFrame()

    try:
        # 인코딩 문제 발생 시 'cp949' or 'euc-kr' 로 변경 시도 필요
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 컬럼 확인
        if text_column not in df.columns:
            print(f"   [Error] CSV 안에 '{text_column}' 컬럼이 없습니다. (현재 컬럼: {list(df.columns)})")
            return pd.DataFrame()
            
        # 필요한 데이터만 추출
        for idx, row in df.iterrows():
            content = str(row[text_column]).strip()
            
            if not content: continue

            data_list.append({
                'ID': f"N_{idx+1:05d}",      # 예: N_00001
                'script': content,           # 대화 내용 전체
                'label': 0,                  # 정상 = 0
                'class': None                # 유형 정보 (보류)
            })
            
        print(f"   -> {len(data_list)}개 데이터 로드 완료.")
        return pd.DataFrame(data_list)
        
    except Exception as e:
        print(f"   [Error] CSV 로드 실패: {e}")
        return pd.DataFrame()

# ==============================================================================
# [메인] 실행 함수
# ==============================================================================
def main():
    print(">>> 데이터셋 통합 작업을 시작합니다...\n")

    # 1. 데이터 로드
    df_phishing = load_phishing_data(CONFIG['PHISHING_DATA_DIR'])
    df_normal = load_normal_data(CONFIG['NORMAL_DATA_PATH'], CONFIG['NORMAL_TEXT_COL'])

    # 2. 데이터 병합
    if df_phishing.empty and df_normal.empty:
        print("\n[Fail] 처리할 데이터가 없습니다. 경로를 확인해주세요.")
        return

    # 두 데이터프레임 합치기
    master_df = pd.concat([df_phishing, df_normal], ignore_index=True)

    # 3. 컬럼 순서 정리 (ID, script, label, class)
    master_df = master_df[['ID', 'script', 'label', 'class']]

    # 4. 데이터 섞기 (Shuffle) - 학습용/테스트용 나눌 때 편하도록 미리 섞음
    master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. 저장
    save_path = CONFIG['OUTPUT_FILE']
    master_df.to_csv(save_path, index=False, encoding='utf-8-sig') # 엑셀 호환을 위해 sig 사용

    print("\n" + "="*50)
    print(f"[Done] 통합 파일 생성 완료!")
    print(f" - 저장 경로: {save_path}")
    print(f" - 전체 데이터 개수: {len(master_df)}")
    print(f" - 피싱(1) 개수: {len(master_df[master_df['label']==1])}")
    print(f" - 정상(0) 개수: {len(master_df[master_df['label']==0])}")
    print("="*50)

    # 샘플 출력
    print("\n[Sample Data]")
    print(master_df.head(3))

if __name__ == "__main__":
    main()