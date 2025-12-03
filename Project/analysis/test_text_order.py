import sys
import os

# [핵심] 현재 파일의 위치를 기준으로 상위 폴더(Project Root)를 찾아서 sys.path에 추가
# 이렇게 하면 'src' 폴더를 import 할 수 있게 됩니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 이제 src를 문제없이 불러올 수 있습니다.
from src.pipeline.preprocessor import ThreeWindowPreprocessor

def test_data_order():
    # 1. 전처리기 초기화
    preprocessor = ThreeWindowPreprocessor(model_name='klue/roberta-base')
    
    print(f"사용 중인 구분자: {preprocessor.sep_token}")

    # 2. 가상 데이터
    fake_data = {
        'prev2': "1. 아주 옛날 이야기",  # t-2
        'prev1': "2. 조금 전 이야기",    # t-1
        'curr':  "3. 지금 하고픈 이야기" # t
    }

    # 3. 전처리 실행
    encoding = preprocessor(
        curr=fake_data['curr'],
        prev1=fake_data['prev1'],
        prev2=fake_data['prev2']
    )

    # 4. 디코딩 확인
    decoded_text = preprocessor.tokenizer.decode(encoding['input_ids'][0], skip_special_tokens=False)

    print("\n" + "="*50)
    print("[검증 결과] 모델 입력 데이터 순서 확인")
    print("="*50)
    print(f"모델 입력: {decoded_text}")
    print("="*50)

    # 5. 검증 로직
    idx_1 = decoded_text.find("1.")
    idx_2 = decoded_text.find("2.")
    idx_3 = decoded_text.find("3.")

    if idx_1 < idx_2 < idx_3:
        print("✅ 성공: (t-2) -> (t-1) -> (t) 순서가 정확합니다.")
    else:
        print("❌ 실패: 문장 순서가 뒤섞였습니다!")

if __name__ == "__main__":
    test_data_order()