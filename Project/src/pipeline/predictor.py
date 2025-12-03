# 임시
import torch
from collections import deque
from src.utils import make_window_text  # [Import] 공통 함수 가져오기

class RealTimePredictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sep = tokenizer.sep_token
        self.model.eval()
        
        # 최근 대화 2개를 저장하는 큐
        self.buffer = deque(maxlen=2) 

    def predict(self, new_sentence):
        # 1. 버퍼 상태 확인하여 prev2, prev1 추출
        history = list(self.buffer)
        
        # 버퍼에 2개 있으면: [0]=prev2, [1]=prev1
        # 버퍼에 1개 있으면: [0]=prev1, prev2는 None
        # 버퍼가 비었으면: prev1, prev2 둘 다 None
        
        if len(history) == 2:
            prev2, prev1 = history[0], history[1]
        elif len(history) == 1:
            prev2, prev1 = None, history[0]
        else:
            prev2, prev1 = None, None
            
        # 2. [핵심] 공통 함수 사용하여 텍스트 생성
        # None을 넘기면 알아서 [START]로 바꿔줍니다.
        text = make_window_text(
            prev2=prev2,
            prev1=prev1,
            curr=new_sentence,
            sep_token=self.sep
        )
        
        # 3. 모델 예측 (토크나이징 및 Forward)
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=128
        )
        
        # GPU 처리 등...
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            logits = outputs['logits'] # model.py 반환값 키에 맞춤
            probs = torch.softmax(logits, dim=1)
            
        # 4. 버퍼 업데이트 (현재 문장을 과거 기록으로 저장)
        self.buffer.append(new_sentence)
        
        # 피싱 확률(Class 1) 반환
        return probs[0][1].item()