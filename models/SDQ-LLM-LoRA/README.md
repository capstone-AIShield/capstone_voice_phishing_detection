---
base_model: skt/kogpt2-base-v2
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:skt/kogpt2-base-v2
- lora
- transformers
---

# SDQ-LLM (Voice Phishing Countermeasure Model)

이 모델은 보이스피싱(전기통신금융사기) 상황을 인식하고, 적절한 대처 방안 및 가이드를 출력하기 위해 훈련된 한국어 기반의 언어 모델 어댑터(LoRA)이다.

## 모델 개요 (Model Details)

### 모델 설명 (Model Description)
이 모델은 `skt/kogpt2-base-v2` 모델을 백본으로 사용하여, 온디바이스(On-device) 환경에서도 가볍게 동작할 수 있도록 설계되었다. 사용자가 처한 보이스피싱 의심 상황(예: 기관 사칭, 카드 배송원 사칭, 가족 납치 빙자 등)을 입력받으면, 즉각적인 행동 요령과 참고해야 할 문서 내용을 포함한 스크립트를 반환한다.

- **개발 목적:** 보이스피싱 피해 최소화를 위한 실시간 대응 가이드라인 제공
- **모델 타입:** Causal Language Model (LoRA 파인튜닝)
- **언어:** 한국어 (ko)
- **파인튜닝 베이스 모델:** `skt/kogpt2-base-v2`

## 사용 방법 (Uses)

### 직접 사용 (Direct Use)
보이스피싱 모니터링 시스템의 핵심 생성부(Generator)로 사용되며, 음성 인식 모델이나 인텐트 분류기(Encoder)의 결과를 입력받아 최종 대응 스크립트를 생성한다.

입력 프롬프트 예시:
```text
명령어: 다음은 카드 배송원 사칭형 보이스피싱 의심 상황이다. 적절한 대처 방안을 제시하라.
답변:
```

출력 예시 포맷:
```text
[출처가 불분명한 문자나 전화번호는 무시하세요] | [카드 배송원 사칭 시나리오 대처법] | [상세 내용]
```

## 훈련 세부 정보 (Training Details)

### 훈련 데이터 (Training Data)
- **출처:** 금융감독원(`fss.or.kr`) 및 보이스피싱지킴이(`counterscam112.go.kr`)의 실제 피해 예방 가이드라인
- **데이터 구성:** 총 50여 개의 보이스피싱 시나리오-대응 쌍(Instruction-Output Pair)으로 합성된 JSONL 데이터셋 사용

### 훈련 하이퍼파라미터 (Training Hyperparameters)
- **LoRA 랭크 (r):** 8
- **LoRA 알파 (lora_alpha):** 32
- **드롭아웃 (lora_dropout):** 0.1
- **Epochs:** 1
- **Batch Size:** 4
- **Gradient Accumulation Steps:** 2
- **Learning Rate:** 3e-4
- **Precision:** FP32 (CPU 훈련 기준)

## 라이브러리 및 환경 (Framework versions)
- PEFT 0.17.1
- Transformers (최신 호환버전)
- PyTorch >= 2.1.0
- Datasets 4.5.0