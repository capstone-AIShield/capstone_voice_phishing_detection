import torch

class Config:
    def __init__(self):
        # 1. 경로 설정
        self.data_path = "data/train_final.csv"     # 학습 데이터 경로
        self.model_save_dir = "models/"             # 모델 저장 폴더
        self.model_name = "klue/roberta-base"       # 사용할 Pre-trained 모델
        
        # 2. 하이퍼파라미터 (튜닝 대상)
        self.num_classes = 2        # 0: 일반, 1: 피싱
        self.epochs = 3             # 전체 데이터를 몇 번 볼 것인가
        self.batch_size = 32        # 한 번에 처리할 데이터 양
        self.learning_rate = 2e-5   # 학습률 (RoBERTa는 보통 1e-5 ~ 5e-5 사용)
        self.max_len = 128          # 토큰 최대 길이 (분석 결과에 따라 80 등으로 수정 가능)
        
        # 3. 시스템 설정
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 2        # DataLoader 프로세스 수 (Windows면 0 추천)

    def print_config(self):
        """현재 설정값을 출력합니다."""
        print(f"\n[Configuration]")
        print(f" - Model: {self.model_name}")
        print(f" - Device: {self.device}")
        print(f" - Epochs: {self.epochs}")
        print(f" - Batch Size: {self.batch_size}")
        print(f" - Learning Rate: {self.learning_rate}")
        print(f" - Max Length: {self.max_len}\n")

# 전역에서 편하게 쓰기 위해 인스턴스 생성
cfg = Config()