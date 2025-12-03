import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments

from src.dataset import ThreeWindowDataset
from src.model import MultiTaskAXModel
from src.trainer import MultiTaskTrainer

def main():
    # =========================================================================
    # 1. 환경 설정 (Configuration)
    # =========================================================================
    config = {
        "model_name": "skt/A.X-Encoder-base",
        "data_path": "dataset.csv",
        "output_dir": "./results",
        "max_len": 128,
        "batch_size": 32,
        "epochs": 3,
        "lr": 2e-5,
        "seed": 42
    }

    # 랜덤 시드 고정 (재현성 확보)
    torch.manual_seed(config["seed"])

    # gpu 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 사용할 디바이스: {device}")

    # =========================================================================
    # 2. 데이터 로드 및 전처리 (Data Loading)
    # =========================================================================
    print(">>> 데이터를 로드합니다...")
    if not os.path.exists(config["data_path"]):
        raise FileNotFoundError("dataset.csv 파일이 없습니다. 전처리 코드를 먼저 실행해주세요.")

    df = pd.read_csv(config["data_path"])

    # Train / Validation 분리 (Stratified Split)
    train_df, val_df = train_test_split(
        df, 
        test_size=0.1, 
        random_state=config["seed"], 
        stratify=df['label']
    )

    print(f"    - Train Size: {len(train_df)}")
    print(f"    - Valid Size: {len(val_df)}")

    # =========================================================================
    # 3. 데이터셋 및 모델 준비 (Dataset & Model Preparation)
    # =========================================================================
    print(">>> 토크나이저와 모델을 로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    train_dataset = ThreeWindowDataset(train_df, tokenizer, max_len=config["max_len"])
    val_dataset = ThreeWindowDataset(val_df, tokenizer, max_len=config["max_len"])

    # 모델 초기화
    model = MultiTaskAXModel(config["model_name"])
    model.to(device)

    # =========================================================================
    # 4. 학습 파라미터 및 트레이너 설정 (Trainer Setup)
    # =========================================================================
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        weight_decay=0.01,
        
        # 로깅 및 저장 설정
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",        # 매 에폭마다 검증
        save_strategy="epoch",        # 매 에폭마다 저장
        load_best_model_at_end=True,  # 가장 성능 좋은 모델 자동 로드
        metric_for_best_model="eval_loss",
        save_total_limit=2,           # 디스크 용량 절약을 위해 최근 2개만 저장
        
        # 기타 최적화 옵션
        dataloader_num_workers=2,     # 데이터 로딩 속도 향상 (Windows는 0 권장)
        fp16=torch.cuda.is_available() # GPU 사용 시 혼합 정밀도(16bit) 사용 (속도 UP, 메모리 절약)
    )

    # Trainer 초기화
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # =========================================================================
    # 5. 학습 실행 (Training Execution)
    # =========================================================================
    print(">>> 학습을 시작합니다.. (Start Training)")
    trainer.train()

    # =========================================================================
    # 6. 최종 모델 저장 (Saving)
    # =========================================================================
    final_save_path = os.path.join(config["output_dir"], "final_model")
    
    print(f">>> 학습 완료. 모델을 '{final_save_path}'에 저장합니다.")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    # [TODO: Multi-Task Expansion] 필요 시 별도 헤드 가중치 저장 로직 추가 가능

if __name__ == "__main__":
    main()