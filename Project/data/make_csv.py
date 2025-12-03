import os
import pandas as pd
from tqdm import tqdm  # 진행률 표시 (없으면 pip install tqdm)

def txt_to_csv(source_folder, output_file):
    # 1. 데이터를 담을 리스트 생성
    data_list = []
    
    # 2. 폴더 내 파일 목록 가져오기
    # os.path.exists 체크
    if not os.path.exists(source_folder):
        print(f"❌ 오류: '{source_folder}' 폴더가 존재하지 않습니다.")
        return

    file_list = [f for f in os.listdir(source_folder) if f.endswith('.txt')]
    print(f"📂 '{source_folder}' 폴더에서 {len(file_list)}개의 텍스트 파일을 찾았습니다.")

    # 3. 파일 하나씩 읽어서 리스트에 추가
    for file_name in tqdm(file_list, desc="파일 병합 중"):
        file_path = os.path.join(source_folder, file_name)
        
        try:
            # 텍스트 파일 읽기 (인코딩 에러 방지를 위해 utf-8 시도 후 cp949 시도)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
            except UnicodeDecodeError:
                # 윈도우 메모장 등으로 저장한 경우 cp949일 수 있음
                with open(file_path, 'r', encoding='cp949') as f:
                    content = f.read().strip()

            # ID는 파일명(확장자 제외)으로 설정
            file_id = os.path.splitext(file_name)[0]
            
            # 데이터 추가 (Label은 요청하신 대로 전부 1)
            data_list.append({
                'id': file_id,
                'script': content,
                'label': 1
            })
            
        except Exception as e:
            print(f"⚠️ '{file_name}' 처리 중 오류 발생: {e}")

    # 4. 데이터프레임 생성
    df = pd.DataFrame(data_list)
    
    # 5. CSV로 저장
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 변환 완료! 저장된 파일: {output_file}")
    print(f"   총 데이터 개수: {len(df)}")
    
    # 미리보기 출력
    print("\n[데이터 미리보기]")
    print(df.head())

if __name__ == "__main__":
    # 경로 설정 (사용자 환경에 맞게 수정 가능)
    #SOURCE_DIR = "data/text_data"   # 텍스트 파일들이 있는 폴더
    SOURCE_DIR = "data/cleaned_text_data"   # 정제된 텍스트 파일들이 있는 폴더
    #OUTPUT_CSV = "data/phishing_data.csv" # 결과로 나올 CSV 파일 경로
    OUTPUT_CSV = "data/phishing_data_v2.csv" # 결과로 나올 CSV 파일 경로
    
    txt_to_csv(SOURCE_DIR, OUTPUT_CSV)