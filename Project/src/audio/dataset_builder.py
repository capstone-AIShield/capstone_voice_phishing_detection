import os
import json
import re
from tqdm import tqdm
from audio_processor import AudioProcessor

class DatasetBuilder:
    """
    폴더 단위 오디오 파일 처리 + txt 파일 생성 + 메타데이터 상세 집계
    """
    def __init__(self, base_dir, input_folders, output_dir, t5_correct=True):
        self.base_dir = base_dir
        self.input_folders = input_folders
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.t5_correct = t5_correct
        self.processor = AudioProcessor()

    def build_dataset(self):
        # 메타데이터 구조 초기화
        metadata = {
            "overview": {
                "total_data_count": 0,      # 총 오디오 파일 개수
                "total_sentence_count": 0   # 총 추출된 문장 개수
            },
            "folders": {}
        }
        
        global_index = 1

        for folder_name in self.input_folders:
            folder_path = os.path.join(self.base_dir, folder_name)
            if not os.path.exists(folder_path):
                print(f"⚠️ 폴더 없음: {folder_name}")
                continue

            file_list = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp3", ".wav"))]
            
            # 파일명을 숫자 기준으로 정렬 (Natural Sort)
            file_list.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

            folder_sentence_sum = 0
            folder_file_details = [] # 각 파일별 상세 정보를 담을 리스트
            
            start_index = global_index

            for f in tqdm(file_list, desc=f"Processing {folder_name}"):
                audio_path = os.path.join(folder_path, f)
                
                # 저장될 파일 이름: 전역 인덱스 사용 (예: 1.txt, 101.txt)
                txt_filename = f"{global_index}.txt"
                txt_path = os.path.join(self.output_dir, txt_filename)

                sentences = []

                # --- 이어하기(Resume) 로직 ---
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as rf:
                        sentences = [line.strip() for line in rf.readlines() if line.strip()]
                else:
                    sentences = self.processor.process_file(audio_path, t5_correct=self.t5_correct)
                    with open(txt_path, "w", encoding="utf-8") as wf:
                        wf.write("\n".join(sentences))

                # 현재 파일의 문장 개수
                cnt = len(sentences)
                folder_sentence_sum += cnt
                
                # 파일별 상세 정보 기록
                folder_file_details.append({
                    "index": global_index,          # 저장된 파일 번호 (예: 101)
                    "original_name": f,             # 원본 파일명 (예: audio_5.wav)
                    "sentence_count": cnt           # 추출된 문장 수
                })

                global_index += 1

            # --- 폴더별 메타데이터 정리 ---
            end_index = global_index - 1
            file_count_in_folder = end_index - start_index + 1
            
            metadata["folders"][folder_name] = {
                "range_str": f"{start_index} ~ {end_index}", # 보기 편한 범위 문자열
                "start_index": start_index,
                "end_index": end_index,
                "file_count": file_count_in_folder,
                "folder_total_sentences": folder_sentence_sum,
                "files": folder_file_details  # 파일별 상세 리스트 포함
            }

            # 전체 통계 업데이트
            metadata["overview"]["total_data_count"] += file_count_in_folder
            metadata["overview"]["total_sentence_count"] += folder_sentence_sum

        # 메타데이터 JSON 저장
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        print(f"\n✅ 데이터셋 완성.")
        print(f"   - 총 파일 수: {metadata['overview']['total_data_count']}")
        print(f"   - 총 문장 수: {metadata['overview']['total_sentence_count']}")
        print(f"   - 메타데이터 저장됨: {metadata_path}")