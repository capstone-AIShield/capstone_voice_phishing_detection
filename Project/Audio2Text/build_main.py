# build_main.py
import torch
import config as CONFIG  # [핵심] config.py 불러오기
from dataset_builder import DatasetBuilder

def main():
    # GPU 체크 (단순 정보 출력용)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- [System] Running on {device.upper()} ---")

    # 설정 정보 출력
    print(f"--- [Config] Settings ---")
    print(f"    BASE_DIR: {CONFIG.BASE_DIR}")
    print(f"    INPUT_FOLDERS: {CONFIG.INPUT_FOLDERS}")
    print(f"    CSV_PATH: {CONFIG.CSV_PATH}")
    print(f"    MODEL: {CONFIG.WHISPER_MODEL_SIZE}")
    
    # [핵심] Config 객체를 통째로 전달
    builder = DatasetBuilder(CONFIG)
    
    # 실행
    try:
        builder.build_dataset()
        print("\n🎉 [Main] All processes completed successfully.")
        
    except KeyboardInterrupt:
        print("\n🛑 [Main] Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ [Main] Critical Error: {e}")

if __name__ == "__main__":
    main()