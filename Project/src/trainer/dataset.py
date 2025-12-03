import torch
from torch.utils.data import Dataset
# 위에서 만든 Preprocessor 클래스 임포트
from src.pipeline.preprocessor import ThreeWindowPreprocessor

class ThreeWindowDataset(Dataset):
    """
    DataLoader가 사용할 데이터셋 클래스.
    내부적으로 Preprocessor를 호출하여 일관된 전처리를 수행합니다.
    """
    def __init__(self, df, model_name='klue/roberta-base', max_len=128):
        self.data = df
        # 전처리기 인스턴스 생성 (여기서 토크나이저도 같이 로드됨)
        self.preprocessor = ThreeWindowPreprocessor(model_name, max_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 데이터 한 행 가져오기
        row = self.data.iloc[idx]
        
        # 2. 전처리기 호출 (Raw Text -> Tensor)
        encoding = self.preprocessor(
            curr=row['curr'],
            prev1=row.get('prev1'),  # 컬럼이 없거나 값이 없으면 None 처리됨
            prev2=row.get('prev2')
        )

        # 3. 차원 정리 및 결과 반환
        # encoding['input_ids']는 [1, 128] 형태이므로 [128]로 만듦 (squeeze)
        # DataLoader가 나중에 batch_size만큼 묶어서 [32, 128]로 만듭니다.
        return {
            'input_ids': encoding['input_ids'].squeeze(0),      
            'attention_mask': encoding['attention_mask'].squeeze(0), 
            'labels': torch.tensor(row['label_class'], dtype=torch.long)
        }