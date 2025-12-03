# 공통으로 사용할 함수 모음

def make_window_text(prev2, prev1, curr, sep_token, start_marker="[START]"):
    """
    3개의 문장(t-2, t-1, t)을 모델 입력 형식으로 결합하는 공통 함수.
    
    Args:
        prev2 (str or None): 전전 문장. 없으면 None 또는 빈 문자열.
        prev1 (str or None): 직전 문장.
        curr (str): 현재 문장 (타겟).
        sep_token (str): 모델의 구분자 토큰 (예: [SEP], </s>)
        start_marker (str): 대화 시작을 알리는 토큰. 기본값 "[START]"

    Returns:
        str: 결합된 하나의 문자열
        예: "[START] [SEP] 안녕 [SEP] 반가워"
    """
    
    # 1. 전전 문장(prev2)
    # None이거나, 빈 문자열이거나, 전처리 파일의 '[더미]' 표시면 -> [START]
    if prev2 is None or str(prev2).strip() == "" or str(prev2) == "[더미]":
        clean_prev2 = start_marker
    else:
        clean_prev2 = str(prev2).strip()

    # 2. 직전 문장(prev1)
    if prev1 is None or str(prev1).strip() == "" or str(prev1) == "[더미]":
        clean_prev1 = start_marker
    else:
        clean_prev1 = str(prev1).strip()

    # 3. 현재 문장(curr)
    clean_curr = str(curr).strip()

    # 4. 문자열 결합 (이 포맷이 학습과 추론에서 동일하게 적용됨)
    # 구조: (t-2) [SEP] (t-1) [SEP] (t)
    combined_text = f"{clean_prev2} {sep_token} {clean_prev1} {sep_token} {clean_curr}"

    return combined_text