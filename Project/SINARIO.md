모델 파이프라인 설계

1. 오디오 처리부[개선중]
- 모델 : faster-whisper-large-v3-turbo
- 전처리 전략 : 
  1. VAD(Voice Activity Detection): Silero VAD 등을 사용하여 무음 구간을 먼저 필터링
  2. Audio Buffer: 15초 길이의 큐(Queue)를 유지
  3. Step: 매 5초마다 큐를 갱신합니다. (0~15초 인식 -> 5~20초 인식)

2. NLP Classification
- Baseline: KLUE-RoBERTa(기본)
- Challenger: ModernBERT(최신) -> 한국어 버전 모델을 활용
- Input: Whisper가 변환한 텍스트(Token Dropout 적용)
- Augmentation: 텍스트 일부를 삭제하거나, 유사 발음 단어로 치환하여 Whisper의 에러를 시뮬레이션한 데이터로 학습.
- 누적 점수 시스템(Leaky Bucket) 도입: 게임에서 데미지를 입으면 체력이 깎이고, 가만히 있으면 체력이 회복되는 원리와 같음.
  1. 초기 위험 점수: 0점 (최대 100점)
  2. 입력 처리 (매 5초마다): 
    - 모델이 **"피싱(High Risk)"**으로 판단 ($\text{Prob} > 0.8$): 점수 +20점
    - 모델이 **"의심(Medium Risk)"**으로 판단 ($0.5 < \text{Prob} \le 0.8$): 점수 +10점
    - 모델이 **"정상(Normal)"**으로 판단 ($\text{Prob} \le 0.5$): 점수 -10점 (단, 0점 미만으로는 안 내려감)
  3. 경고 단계 (Threshold): 
    - 점수 $\ge$ 30점: [주의] 진동 1회, 화면에 노란색 경고등. (약 2~3회 연속 탐지 시 발동)
    - 점수 $\ge$ 60점: [경고] 진동 연속, 붉은색 화면 점멸, "통화를 종료하세요" 문구. (약 3~4회 연속 강력 탐지 시 발동)
- 보이스 피싱 주의/경고 시 대처 방안 제안: 대화 내용을 바탕으로 어떤 내용인지 분석하여 그 내용에 맞는 대처 방안을 제시, 단 일반 통화 시에는 추론 안 함.
- Architecture: 모델 레이어 구성
  1. Backbone (RoBERTa)
  2. Task Head (Classifier)
  3. [선택](문맥 파악 후 문맥에 따른 대처 방안 출력) -> 별도의 모델로 구성할 수도 있음. 혹은 하나의 모델 내에서 새로운 레이어를 추가해서 동시에 결과를 출력하는 모델을 구성할 수도 있음.

3. 데이터 셋 전처리(학습용)
  1. 컬럼 종류: ID, script, label, class
    - script: 하나의 통화 내용에 대한 대화 정보 전체 (전처리 전 원본 텍스트)
    - label: 보이스 피싱 데이터 인지 아닌지에 대한 정보 (보이스 피싱: 1, 일반 대화: 0)
    - class: 보이스 피싱의 유형 정보 [보류] -> 아직 정리 안 됨.
  2. 텍스트 데이터 증강 기법 사용
    - 구조적 증강 (Structural Augmentation): 슬라이딩 윈도우 (Sliding Window)[기본]
      1. 방식: 전체 대본을 KSS 라이브러리로 문장 단위 분리 후, 5문장씩 묶어서 입력 데이터로 생성.
      2. 설정: Window Size = 5 (입력 길이), Stride = 2 (2문장씩 이동하며 중복 생성).
    - 내용적 증강 (Content Augmentation): KoEDA (Korean Easy Data Augmentation)
      - Whisper(STT) 모델의 인식 오류(오타, 누락, 순서 뒤바뀜 등)에 강건한(Robust) 모델을 만들기 위함.
      - 적용 방식: 학습(Train) 단계에서만 실시간(On-the-fly)으로 확률적으로 적용.
      1. SR (Synonym Replacement): 유의어 교체 (문맥 이해도 향상)
      2. RI (Random Insertion): 무작위 단어 삽입 (추임새 및 STT 노이즈 대비)
      3. RS (Random Swap): 단어 위치 교환 (어순 도치 및 인식 오류 대비)
      4. RD (Random Deletion): 무작위 단어 삭제 (Whisper의 단어 누락/묵음 처리 대비 핵심)
  3. 데이터 분할 전략 (Data Splitting)
    - 그룹 기반 분할 (Group-based Split): 슬라이딩 윈도우 적용 전, 파일 ID(통화 건)를 기준으로 학습용과 테스트용을 분리.

4. 지식 증류 방법(DistilBERT 방법 차용)
  - Teacher Model (선생님): 12-Layer Transformer Encoder
  - Student Model (학생): 6-Layer Transformer Encoder (경량화 모델)
  - 손실 함수 구성 (Loss Function Design)
    1. Hard Loss (Cross Entropy Loss): 학생 모델이 **실제 정답(Label: 0 or 1)**을 정확히 맞추는지 평가.
    2. Soft Loss (Knowledge Distillation Loss - KL Divergence): 선생님 모델이 출력하는 **확률 분포(Soft Label)**를 학생이 모방하도록 유도. 선생님이 느끼는 '유사함'이나 '불확실성' 정보까지 학습.
      - Temperature ($T=2.0$) 스케일링을 적용하여 확률 분포를 부드럽게 만들어 정보량을 극대화.
    3. Cosine Embedding Loss (Hidden State Alignment): 선생님과 학생의 [CLS] 토큰(문장 전체 의미 벡터)의 방향성을 일치시킴.


## 해야할 일 정리
1. KLUE-RoBERTa 모델로 실험 진행 [지훈]
2. ModernBERT(한국어 전용) 모델로 실험 진행[나현]
3. 문맥 파악 후 문맥에 따른 대처 방안 출력을 위해 추가 모델을 만들 것인지 하나의 모델로 통합할 것인지 고민 필요
4. 실시간 동작을 위한 웹 사이트 개발 진행 필요[수효]
5. 오디오 데이터 입력 부터 추론까지 가능한 파이프라인 구현 필요
6. 오디오 데이터 텍스트 변환 코드 개선 필요[지훈]