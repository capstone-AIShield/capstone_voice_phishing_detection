import torch
import torch.nn as nn
from transformers import Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def compute_metrics(p):
    """
    Validation 단계에서 성능을 평가하기 위한 함수입니다.
    Trainer에 인자로 전달됩니다.
    """
    predictions, labels = p
    
    # 모델 출력(predictions)이 튜플이나 딕셔너리일 경우 로짓을 추출
    # MultiTaskRoBERTaModel은 dict를 반환하므로 logits 키나 첫 번째 요소를 가져옵니다.
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    elif isinstance(predictions, dict):
        predictions = predictions["logits"]
        
    # Logits -> Class ID 변환 (Softmax 대신 argmax 사용)
    preds = np.argmax(predictions, axis=1)
    
    # 지표 계산 (Binary Classification 기준)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

class MultiTaskTrainer(Trainer):
    """
    Hugging Face Trainer를 상속받아 커스텀 Loss를 계산하는 클래스.
    MultiTaskRoBERTaModel과 ThreeWindowDataset 사이를 연결합니다.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ---------------------------------------------------------------------------
        # 1. 정답 라벨 가져오기
        # ---------------------------------------------------------------------------
        # dataset.py에서 'labels'라는 키로 텐서를 반환하도록 설정했으므로 이를 가져옵니다.
        # 만약 dataset.py를 'labels_cls'로 수정했다면 inputs.get("labels_cls")를 써야 합니다.
        labels = inputs.get("labels") 

        # ---------------------------------------------------------------------------
        # 2. 모델 실행 (Forward)
        # ---------------------------------------------------------------------------
        # inputs 딕셔너리를 언패킹(**inputs)하여 모델에 전달
        outputs = model(**inputs)

        # model.py에서 반환한 딕셔너리에서 'logits'를 꺼냅니다.
        # (사용자 코드의 logits_cls -> logits 로 매핑)
        logits = outputs.get("logits")

        # ---------------------------------------------------------------------------
        # 3. 손실 함수 적용 (CrossEntropyLoss)
        # ---------------------------------------------------------------------------
        # [Task A] 분류: CrossEntropyLoss
        loss_fct = nn.CrossEntropyLoss()

        try:
            if labels is None:
                raise ValueError("Dataset에서 'labels'를 찾을 수 없습니다.")
            if logits is None:
                raise ValueError("Model 출력에서 'logits'를 찾을 수 없습니다.")

            # 차원 맞추기
            # logits: (Batch_Size, 2)
            # labels: (Batch_Size)
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        except Exception as e:
            print(f"\n⚠️ Loss 계산 중 오류 발생: {e}")
            print(f" - Labels shape: {labels.shape if labels is not None else 'None'}")
            print(f" - Logits shape: {logits.shape if logits is not None else 'None'}")
            raise e

        # ---------------------------------------------------------------------------
        # 4. 결과 반환
        # ---------------------------------------------------------------------------
        # 현재는 Single Task이므로 loss가 곧 total_loss입니다.
        # 나중에 loss_reg(회귀)가 추가되면 total_loss = loss_cls + 0.5 * loss_reg 등으로 확장하세요.
        total_loss = loss

        return (total_loss, outputs) if return_outputs else total_loss