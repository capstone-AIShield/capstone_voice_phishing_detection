import torch.nn as nn
from transformers import AutoModel

class AXBackbone(nn.Module):
    """
    Hugging Face의 사전학습된 모델(skt/A.X-Encoder-base)을 로드하여
    입력 문장(또는 문맥)의 특징 벡터(Embedding)를 추출하는 역할
    """
    def __init__(self, model_name='skt/A.X-Encoder-base'):
        super(AXBackbone, self).__init__()

        # 1. Hugging Face AutoModel로 로드
        self.bert = AutoModel.from_pretrained(model_name)

        # 2. 출력 차원 크기 저장 (보통 768)
        # 나중에 Head를 붙일 때 입력 차원을 알기 위해서 필요
        self.output_dim = self.bert.config.hidden_size
    
    def forward(self, input_ids, attention_mask):
        # 3. BERT 모델에 입력 문장 전달
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [중요] pooler_output 사용
        # BERT의 [CLS] 토큰에 해당하는 벡터로, 문장 전체의 의미를 함축하고 있습니다.
        # Shape: (Batch_Size, 768)
        return outputs.pooler_output
    
class RiskLevelTaskHead(nn.Module):
    """
    백본에서 나온 특징 벡터를 받아서 최종 클래스로 분류하는 신경망
    (Linear -> BN -> ReLU -> Dropout -> Linear 구조)
    """
    def __init__(self, input_dim, num_classes):
        super(RiskLevelTaskHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
    
"""
추가 Task Head 추가 사용 가능
"""
    
class MultiTaskAXModel(nn.Module):
    """
    Backbone과 Head를 결합한 최종 모델 클래스.
    현재는 '3-Window Phishing Detection' 태스크 하나만 있지만,
    추후 self.head_emotion 등을 추가하여 멀티 태스크로 확장이 용이함.
    """
    def __init__(self, model_name='skt/A.X-Encoder-base'):
        super(MultiTaskAXModel, self).__init__()

        # 1. Backbone 초기화
        self.backbone = AXBackbone(model_name=model_name)
        hidden_dim = self.backbone.output_dim # 768

        # 2. 태스크별 Head 초기화
        self.head_RiskLevel = RiskLevelTaskHead(input_dim=hidden_dim, num_classes=3)

        # 3. 추가 태스크 Head도 여기에 초기화 가능

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: 토큰화된 입력 ID
            attention_mask: 어텐션 마스크
            labels: (Trainer 호환용) 사용하지 않더라도 인자로 받아줌
        """
        # 1. Backbone 통해 특징 벡터 추출
        shared_features = self.backbone(input_ids, attention_mask)

        # 2. 태스크별 Head로 분류 수행
        logits = self.head_RiskLevel(shared_features)

        # 확장 예시
        # emotion_logits = self.head_emotion(shared_features)

        # 3. 결과 반환
        return {
            'logits': logits,
            # 'emotion_logits': emotion_logits, # 확장 시 추가
        }