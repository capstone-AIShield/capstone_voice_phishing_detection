# 데이터를 csv 파일로 저장하는 유틸리티 함수
import os
import re
import csv
import pandas as pd
from tqdm import tqdm
from audio_processor import AudioProcessor

class DatasetBuilder:
    """
    오디오 -> 텍스트 -> CSV 실시간 저장
    * Config 설정을 사용하여 경로 및 옵션을 관리합니다.
    * Future-proof: 나중에 'class' 정보가 추가될 것을 대비해 검증 로직을 유연하게 설계했습니다.
    """
    def __init__(self, config):
        """
        config.py의 설정을 받아 초기화합니다.
        """
        # 1. 경로 설정 (config.py에서 가져옴)
        self.base_dir = config.BASE_DIR
        self.input_folders = config.INPUT_FOLDERS
        self.output_dir = config.OUTPUT_DIR
        self.csv_path = config.CSV_PATH
        
        # 2. 라벨 및 클래스 매핑 규칙 (config.py에서 가져옴)
        self.folder_label_map = config.FOLDER_LABEL_MAP
        self.folder_class_map = config.FOLDER_CLASS_MAP
        
        # 메타데이터/로그 저장용 폴더 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 3. AudioProcessor 초기화
        # (AudioProcessor는 config에 정의된 모델 사이즈를 사용하도록 설정)
        self.processor = AudioProcessor(whisper_model_size=config.WHISPER_MODEL_SIZE)

    def get_existing_progress(self):
        """
        CSV 마지막 행의 무결성을 검사하고 이어하기 지점을 반환합니다.
        * 데이터가 불완전하면(쓰다가 끊김) 해당 행을 삭제하고 다시 작업합니다.
        """
        if not os.path.exists(self.csv_path):
            return 1, set()

        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return 1, set()

            last_idx = df.index[-1]
            last_row = df.iloc[-1]
            
            # [검증 로직] 필수 컬럼이 비어있는지 확인
            # 현재는 ID, script, label만 확인하지만, 나중에 class가 필수라면 여기에 추가 가능
            required_cols = ['ID', 'script', 'label'] 

            is_complete = True
            
            # 1. 필수 컬럼 중 하나라도 비어있으면(NaN) 불완전으로 간주
            for col in required_cols:
                if pd.isna(last_row[col]):
                    print(f"⚠️ 데이터 불완전: {col} 값이 없습니다. (ID: {last_row['ID']})")
                    is_complete = False
                    break
            
            # 2. script가 빈 문자열인지 추가 확인
            if is_complete and str(last_row['script']).strip() == "":
                print(f"⚠️ 데이터 불완전: script 내용이 없습니다. (ID: {last_row['ID']})")
                is_complete = False

            # --- 불량 데이터 처리 ---
            if not is_complete:
                print(f"🔄 마지막 행(ID: {last_row['ID']})을 삭제하고 해당 파일부터 다시 시작합니다.")
                df = df.drop(last_idx)
                # 수정된 내용을 다시 저장 (utf-8-sig: 엑셀 한글 호환)
                df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
                
                # 삭제 후 다시 마지막 번호 계산
                if not df.empty:
                    last_id_str = df['ID'].iloc[-1]
                    next_index = int(last_id_str.split('_')[1]) + 1
                else:
                    next_index = 1
            else:
                # 정상이면 다음 번호 계산
                next_index = int(last_row['ID'].split('_')[1]) + 1
                print(f"✅ 이어하기: ID P_{next_index:04d} 부터 시작합니다.")

            # 처리된 파일 목록 반환 (중복 작업 방지용)
            processed_files = set(df['filename'].astype(str).values)
            return next_index, processed_files

        except Exception as e:
            print(f"⚠️ CSV 읽기 오류 (새로 시작): {e}")
            return 1, set()

    def build_dataset(self):
        print(f"🚀 데이터셋 구축 시작 (Output: {self.csv_path})")
        
        # 1. 이어하기 정보 가져오기
        global_index, processed_files = self.get_existing_progress()

        # 2. CSV 파일 열기 (Append 모드)
        file_exists = os.path.exists(self.csv_path)
        
        # newline='' 옵션은 윈도우에서 줄바꿈이 두 번 되는 것을 방지
        with open(self.csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 파일이 없거나 비어있으면 헤더 작성
            if not file_exists or os.path.getsize(self.csv_path) == 0:
                # [확장성] class 컬럼은 미리 만들어둡니다.
                writer.writerow(['ID', 'script', 'label', 'class', 'filename'])

            for folder_name in self.input_folders:
                folder_path = os.path.join(self.base_dir, folder_name)
                
                if not os.path.exists(folder_path):
                    print(f"⚠️ 폴더 없음: {folder_name}")
                    continue

                # 파일 리스트업 및 숫자 기준 정렬 (1.wav, 2.wav, 10.wav ...)
                file_list = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp3", ".wav"))]
                file_list.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

                # 라벨 및 클래스 결정 (config에 정의된 맵 사용)
                current_label = self.folder_label_map.get(folder_name, 1)
                
                # 나중에 config.py의 FOLDER_CLASS_MAP을 채우면 자동으로 적용됨
                current_class = self.folder_class_map.get(folder_name, None)

                print(f"\n📂 Processing {folder_name} (Label: {current_label}, Class: {current_class})")

                for filename in tqdm(file_list, desc=folder_name):
                    # 이미 처리된 파일이면 건너뜀
                    if filename in processed_files:
                        continue

                    audio_path = os.path.join(folder_path, filename)
                    
                    # --- [핵심] STT 변환 및 전처리 ---
                    sentences = self.processor.process_file(audio_path)
                    full_script = " ".join(sentences).strip()
                    
                    # 유효한 스크립트가 있을 때만 저장
                    if full_script:
                        file_id = f"P_{global_index:04d}"
                        
                        writer.writerow([
                            file_id,
                            full_script,
                            current_label,
                            current_class, # 현재는 비어있음(None)
                            filename
                        ])
                        
                        # [안전장치] 즉시 디스크에 쓰기 (전원 차단 대비)
                        f.flush()
                        
                        global_index += 1
                        
                        # 현재 실행 중 중복 방지를 위해 추가
                        processed_files.add(filename)

        print(f"\n✅ [완료] 모든 작업이 끝났습니다.")
        print(f"   - 저장 경로: {os.path.abspath(self.csv_path)}")