import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    CSV 파일을 읽어 Train / Validation / Test 데이터프레임으로 분할하여 반환합니다.
    
    Args:
        file_path (str): 데이터셋 CSV 경로
        train_ratio (float): 학습 데이터 비율 (기본 0.8)
        val_ratio (float): 검증 데이터 비율 (기본 0.1)
        test_ratio (float): 테스트 데이터 비율 (기본 0.1)
        seed (int): 재현성을 위한 랜덤 시드
        
    Returns:
        (train_df, val_df, test_df): 분할된 3개의 Pandas DataFrame
    """
    
    # 비율 합계 검증
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("비율의 합은 반드시 1.0이어야 합니다. (예: 0.8, 0.1, 0.1)")

    # 1. CSV 로드
    print(f"📂 데이터 로드 중: {file_path}")
    df = pd.read_csv(file_path)
    
    # 2. 1차 분할: Train vs (Val + Test)
    # stratify=df['label_class'] 옵션을 통해 라벨 비율을 유지합니다.
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=df['label_class']
    )
    
    # 3. 2차 분할: Val vs Test
    # 남은 데이터(temp_df)에서 Val과 Test의 비율을 계산
    # 예: 전체의 20%가 남았고, Val:Test가 1:1이라면 0.5로 나눔
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_df['label_class']
    )
    
    print(f"📊 데이터 분할 완료")
    print(f"   - Train Set : {len(train_df):,}개")
    print(f"   - Val Set   : {len(val_df):,}개")
    print(f"   - Test Set  : {len(test_df):,}개")
    
    return train_df, val_df, test_df