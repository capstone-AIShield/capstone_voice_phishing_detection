import torch
from transformers import AutoTokenizer
from src.utils.text_utils import make_window_text

class ThreeWindowPreprocessor:
    """
    3문장(t-2, t-1, t)을 결합하고 토크나이징하여 Tensor로 변환하는 클래스.
    학습(Dataset)과 추론(Inference Pipeline) 양쪽에서 인스턴스를 생성해 사용합니다.
    """
    def __init__(self, model_name='klue/roberta-base', max_len=128):
        self.model_name = model_name
        self.max_len = max_len
        
        # 1. 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 2. Special Token 추가 ([START])
        # 모델이 이 토큰을 인식하게 하려면 추후 모델 임베딩 리사이징이 필요합니다.
        special_tokens = {'additional_special_tokens': ['[START]']}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # 3. 구분자 저장
        self.sep_token = self.tokenizer.sep_token

    def __call__(self, curr, prev1=None, prev2=None):
        """
        Args:
            prev2 (str): 전전 문장 (없으면 None)
            prev1 (str): 직전 문장 (없으면 None)
            curr (str): 현재 문장 (타겟)
            
        Returns:
            dict: {'input_ids': tensor, 'attention_mask': tensor}
        """
        # 1. 텍스트 결합 (Utils 함수 사용)
        full_text = make_window_text(
            prev2=prev2,
            prev1=prev1,
            curr=curr,
            sep_token=self.sep_token,
            start_marker="[START]"
        )

        # 2. 토크나이징 (PyTorch Tensor 반환)
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=True,  # [CLS], [SEP] 포함
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'],       # shape: [1, max_len]
            'attention_mask': encoded['attention_mask'] # shape: [1, max_len]
        }
    
    @property
    def vocab_size(self):
        """변경된(추가된) 단어장 크기를 반환 (모델 resize용)"""
        return len(self.tokenizer)