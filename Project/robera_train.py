import os
import torch
import numpy as np
import random
from transformers import TrainingArguments

# -----------------------------------------------------------------------------
# [모듈 임포트] 요청하신 파일명 및 경로 반영
# -----------------------------------------------------------------------------
from src.core.roberta_config import cfg                  # 설정 (Config)
from src.utils.data_utils import load_and_split_data     # 데이터 분할 Utils
from src.trainer.dataset import ThreeWindowDataset       # 데이터셋 Class (src/trainer/dataset.py)
from src.core.roberta_model import MultiTaskRoBERTaModel # 모델 Class (src/core/roberta_model.py)
from src.trainer.roberta_trainer import MultiTaskTrainer, compute_metrics # 트레이너 (src/trainer/roberta_trainer.py)

def set_seed(seed):
    """재현성을 위해 모든 랜덤 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # -------------------------------------------------------------------------
    # 1. 환경 설정 (System Setup)
    # -------------------------------------------------------------------------
    set_seed(cfg.seed)
    print(f"\n[System] Using Device: {cfg.device}")
    
    # 설정값 출력 (선택 사항)
    cfg.print_config()

    # -------------------------------------------------------------------------
    # 2. 데이터 로드 및 분할 (Data Load & Split)
    # -------------------------------------------------------------------------
    # CSV 파일을 읽어서 8:1:1 비율로 나눕니다.
    print(f"\n[Data] Loading data from {cfg.data_path}")
    train_df, val_df, test_df = load_and_split_data(
        file_path=cfg.data_path,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        seed=cfg.seed
    )

    # -------------------------------------------------------------------------
    # 3. 데이터셋 초기화 (Dataset Initialization)
    # -------------------------------------------------------------------------
    print("[Data] Initializing Datasets and Tokenizer...")
    # Dataset 내부에서 Preprocessor가 초기화되고, [START] 토큰이 추가됩니다.
    train_dataset = ThreeWindowDataset(train_df, model_name=cfg.model_name, max_len=cfg.max_len)
    val_dataset = ThreeWindowDataset(val_df, model_name=cfg.model_name, max_len=cfg.max_len)
    # test_dataset은 추후 평가 시 사용 (필요하면 로드)
    
    # [중요] Preprocessor에서 늘어난 Vocab Size 가져오기
    vocab_size = train_dataset.preprocessor.vocab_size
    print(f"[Info] Updated Vocab Size: {vocab_size} (includes [START] token)")

    # -------------------------------------------------------------------------
    # 4. 모델 초기화 (Model Initialization)
    # -------------------------------------------------------------------------
    print(f"\n[Model] Loading {cfg.model_name}...")
    model = MultiTaskRoBERTaModel(
        model_name=cfg.model_name, 
        num_classes=cfg.num_classes,
        new_vocab_size=vocab_size  # [필수] 임베딩 리사이징
    )
    model.to(cfg.device)

    # -------------------------------------------------------------------------
    # 5. 학습 인자 설정 (Training Arguments)
    # -------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.model_save_dir, "checkpoints"),
        
        # 학습 하이퍼파라미터
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        
        # 최적화 옵션
        fp16=torch.cuda.is_available(),  # GPU 사용 시 Mixed Precision (속도 향상)
        
        # 평가 및 저장 전략
        eval_strategy="epoch",  # 매 Epoch 끝날 때마다 평가
        save_strategy="epoch",        # 매 Epoch 끝날 때마다 모델 저장
        save_total_limit=2,           # 최근 2개 모델만 유지 (용량 절약)
        load_best_model_at_end=True,  # 학습 종료 시 가장 성능 좋은 모델 로드
        metric_for_best_model="f1",   # Best Model 기준: F1 Score
        
        # 로깅
        logging_dir='./logs',
        logging_steps=50,
        seed=cfg.seed,
        dataloader_num_workers=cfg.num_workers
    )

    # -------------------------------------------------------------------------
    # 6. 트레이너 초기화 (Trainer Initialization)
    # -------------------------------------------------------------------------
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # -------------------------------------------------------------------------
    # 7. 학습 실행 (Training)
    # -------------------------------------------------------------------------
    print("\n[Train] Starting Training Loop...")
    trainer.train()

    # -------------------------------------------------------------------------
    # 8. 최종 결과 저장 (Saving)
    # -------------------------------------------------------------------------
    save_path = os.path.join(cfg.model_save_dir, "best_model")
    print(f"\n[Save] Saving best model to {save_path}")
    
    # (1) 모델 가중치 저장
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    
    # (2) 토크나이저 저장 (필수: [START] 토큰 정보 포함됨)
    train_dataset.preprocessor.tokenizer.save_pretrained(save_path)
    
    print("[Done] All processes finished successfully!")

if __name__ == "__main__":
    main()