from dataset_builder import DatasetBuilder

# -------------------- 설정 --------------------
BASE_DIR = "../보이스 피싱 데이터(금감원)"
INPUT_FOLDERS = ["바로 이 목소리", "대출 사기형", "수사기관 사칭형"]
OUTPUT_DIR = "./data/text_data_v3"  # 오디오 파일 단위 txt 저장

T5_CORRECT = True  # T5 한국어 교정 사용 여부

def main():
    builder = DatasetBuilder(BASE_DIR, INPUT_FOLDERS, OUTPUT_DIR, t5_correct=T5_CORRECT)
    builder.build_dataset()

if __name__ == "__main__":
    main()
