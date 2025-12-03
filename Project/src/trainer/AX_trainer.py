import torch.nn as nn
from transformers import Trainer

class MultiTaskTrainer(Trainer):
    """
    Hugging Face Trainer를 상속받아 표준적인 CrossEntropyLoss를 계산하는 클래스.
    커스텀 모델(MultiTaskAXModel)의 출력 형식(Dictionary)에 맞춰 Loss를 연결해줍니다.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ---------------------------------------------------------------------------
        # 1. 정답 라벨 가져오기 (데이터셋에서 두 종류의 라벨을 준다고 가정)
        # ---------------------------------------------------------------------------
        labels_cls = inputs.get("labels_cls")  # 분류용 정답 (0 or 1)
        #labels_reg = inputs.get("labels_reg")  # 회귀용 정답 (예: 위험도 점수)

        # ---------------------------------------------------------------------------
        # 2. 모델 실행 (Forward)
        # ---------------------------------------------------------------------------
        outputs = model(**inputs)

        # 모델이 딕셔너리로 두 개의 로짓을 뱉는다고 가정
        logits_cls = outputs.get("logits_cls") # 분류 헤드의 출력
        #logits_reg = outputs.get("logits_reg") # 회귀 헤드의 출력


        # ---------------------------------------------------------------------------
        # 3. 각각 다른 손실 함수 적용 (핵심!)
        # ---------------------------------------------------------------------------
        # [Task A] 분류: CrossEntropyLoss
        loss_fct_cls = nn.CrossEntropyLoss()

        # [Task B] 회귀: MSELoss (Mean Squared Error)
        # 회귀 값은 보통 float 형이므로 차원과 타입을 맞춰주는 게 중요합니다.
        #loss_fct_reg = nn.MSELoss()
        # logits_reg가 (Batch, 1) 형태라면 squeeze()로 펴줍니다.
        #loss_reg = loss_fct_reg(logits_reg.view(-1), labels_reg.view(-1).float())
        
        try:
            if labels_cls is None or logits_cls is None:
                raise ValueError("labels_cls 또는 logits_cls가 None입니다.")
            loss_cls = loss_fct_cls(logits_cls.view(-1, 2), labels_cls.view(-1))
        except ValueError as e:
            print(f"⚠️ Loss 계산 중 오류 발생: {e}")
            print(f"   - labels_cls 타입: {type(labels_cls)}")
            print(f"   - logits_cls 타입: {type(logits_cls)}")
            raise e

        

        # ---------------------------------------------------------------------------
        # 4. Total Loss 계산 (가중치 적용)
        # ---------------------------------------------------------------------------
        total_loss = (1.0 * loss_cls) # + (0.5 * loss_reg)  # 예시: 분류에 더 큰 비중

        return (total_loss, outputs) if return_outputs else total_loss

