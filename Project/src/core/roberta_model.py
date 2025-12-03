import torch.nn as nn
from transformers import AutoModel

class RoBERTaBackbone(nn.Module):
    """
    Hugging Face의 사전학습된 모델(klue/roberta-base)을 로드하여
    입력 문장(또는 문맥)의 특징 벡터(Embedding)를 추출하는 역할
    
    [수정됨] Token Dropout 기능을 추가하여 문맥 학습 강화
    """
    def __init__(self, model_name='klue/roberta-base', new_vocab_size=None, token_dropout_prob=0.1):
        super(RoBERTaBackbone, self).__init__()

        # 1. Hugging Face AutoModel로 로드
        self.model = AutoModel.from_pretrained(model_name)

        # [필수 추가] Preprocessor에서 추가한 [START] 토큰 반영을 위한 리사이징
        if new_vocab_size is not None:
            print(f"[Model] Resizing token embeddings to {new_vocab_size}")
            self.model.resize_token_embeddings(new_vocab_size)
            
        # [수정됨] Token Dropout 확률 설정 (예: 0.1 = 10%의 토큰을 랜덤하게 0으로 만듦)
        # 이상한 단어가 섞여 있어도 모델이 이를 무시하고 문맥을 보게 유도함
        self.token_dropout_prob = token_dropout_prob

        # 2. 출력 차원 크기 저장 (RoBERTa Base = 768)
        self.output_dim = self.model.config.hidden_size
    
    def forward(self, input_ids, attention_mask):
        # [수정됨] 모델을 바로 통과시키지 않고 임베딩(Embedding) 벡터를 먼저 추출
        # self.model.embeddings는 토큰 ID를 벡터로 변환하는 레이어입니다.
        embedding_output = self.model.embeddings(input_ids=input_ids)

        # [수정됨] 학습(Training) 모드일 때만 Token Dropout 적용
        if self.training and self.token_dropout_prob > 0:
            # 배치 내의 각 토큰에 대해 유지할지(1), 지울지(0)를 결정하는 마스크 생성
            # 베르누이 분포 사용: (1 - p) 확률로 1 생성
            keep_prob = 1 - self.token_dropout_prob
            mask = (torch.rand(input_ids.shape, device=input_ids.device) < keep_prob).float()
            # 마스크 차원 확장: (Batch, Seq_Len) -> (Batch, Seq_Len, 1)
            # 임베딩 벡터의 모든 차원(768)을 동시에 0으로 만들기 위함
            mask = mask.unsqueeze(-1)
            
            # 임베딩에 마스크 적용 (Noise 주입)
            embedding_output = embedding_output * mask

        # [수정됨] 노이즈가 섞인 임베딩을 모델의 인코더에 입력
        # input_ids 대신 inputs_embeds 인자를 사용해야 함
        outputs = self.model(
            inputs_embeds=embedding_output, # ID 대신 임베딩 벡터 전달
            attention_mask=attention_mask
        )

        # [중요] pooler_output 사용
        # RoBERTa의 [CLS] 토큰 벡터 (Batch_Size, 768)
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

class MultiTaskRoBERTaModel(nn.Module):
    """
    Backbone과 Head를 결합한 최종 모델 클래스.
    """
    def __init__(self, model_name='klue/roberta-base', num_classes=2, new_vocab_size=None):
        super(MultiTaskRoBERTaModel, self).__init__()

        # 1. Backbone 초기화 (단어장 크기 전달)
        self.backbone = RoBERTaBackbone(model_name=model_name, new_vocab_size=new_vocab_size)
        hidden_dim = self.backbone.output_dim # 768

        # 2. 태스크별 Head 초기화 (이진 분류: num_classes=2)
        self.head_RiskLevel = RiskLevelTaskHead(input_dim=hidden_dim, num_classes=num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: 토큰화된 입력 ID
            attention_mask: 어텐션 마스크
            labels: (Trainer 호환용)
        """
        # 1. Backbone 통해 특징 벡터 추출
        shared_features = self.backbone(input_ids, attention_mask)

        # 2. 태스크별 Head로 분류 수행
        logits = self.head_RiskLevel(shared_features)

        # 3. 결과 반환 (딕셔너리 형태 유지)
        return {
            'logits': logits,
        }