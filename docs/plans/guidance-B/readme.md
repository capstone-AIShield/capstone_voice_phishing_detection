# 인코더 활용 대처방안 — 실험 설계 계획서

> **대처방안 #3 | On-device AI**
> 출력 품질 향상 · 용량 제약 해결 · 서비스 지연 안정화의 정량적 검증을 위한 실험 프레임워크

---

## 목차

1. [실험 목표 및 가설](#1-실험-목표-및-가설)
2. [실험군 구성](#2-실험군-구성-experimental-groups)
3. [주요 실험 절차](#3-주요-실험-절차-experimental-pipeline)
4. [베이스라인 구성](#4-베이스라인-구성)
5. [평가 지표](#5-평가-지표-evaluation-metrics)
6. [재현성 체크리스트](#6-재현성-체크리스트)
7. [기대 결과](#7-기대-결과-expected-outcomes)
8. [참고 문헌 및 레퍼런스](#8-참고-문헌-및-레퍼런스)
   - [핵심 논문·기술보고서](#핵심-논문기술보고서)
   - [대표 오픈소스·공식 프로젝트](#대표-오픈소스공식-프로젝트)

---

## 1. 실험 목표 및 가설

### 1.1 실험 목표

단순 모델 추론 대비 **인코더–정제 모듈 조합**이 자원 제약 환경(On-device)에서 얼마나 높은 정확도와 효율성을 달성하는지 검증한다.

### 1.2 핵심 가설

| 가설 | 내용 |
|------|------|
| **H1** | 인코더 출력값에 확률 밀도 맵(PDM) 기반 정제 레이어를 추가하면, 모델 크기를 늘리지 않고도 **가양성(False Positive)을 50% 이상** 감소시킬 수 있다. |
| **H2** | 양자화(SigmaQuant) 적용 시 **메모리 점유율을 40% 이상** 절감하면서도 성능 저하를 최소화할 수 있다. |

---

## 2. 실험군 구성 (Experimental Groups)

비교 분석을 위해 다음 세 가지 그룹으로 모델을 구성한다.

| 그룹 | 구성 (Architecture) | 주요 특징 |
|------|---------------------|-----------|
| **A — Baseline** | Vanilla Encoder | BERT-Base 또는 MobileBERT 기반. 정제 과정 없이 소프트맥스 임계값만으로 필터링 적용 |
| **B — Refined** | Lightweight Encoder + NR Module | PDM 특징을 분석하는 Decision Tree 기반 정제 레이어(NR Module) 추가 |
| **C — Optimized** | Refined + SigmaQuant | Group B에 SigmaQuant 적용. 레이어별 민감도에 따라 4-bit/8-bit 가변 양자화를 적용하여 용량 제약 해결 |

---

## 3. 주요 실험 절차 (Experimental Pipeline)

### 단계 1: 데이터셋 준비 및 전처리

도메인 특화 데이터(임상 노트, 원격 탐사 데이터 등)를 활용하여 데이터셋 $\mathcal{D} = \{ (x_i, y_i) \}_{i=1}^N$ 를 구축한다.
공정한 평가를 위해 학습/검증/테스트셋을 **70:15:15** 비율로 계층적 분할(Stratified Splitting)하여 클래스 불균형에 의한 편향을 방지한다.

| 데이터셋 | 활용 목적 | 주요 특징 |
|----------|-----------|-----------|
| **BEIR** | 범용 검색·재랭킹 | 제로샷 IR 벤치마크, 18개 이종 도메인 |
| **MS MARCO** | 대규모 검색·QA | 대규모 정보 검색 및 질의응답 |
| **MIRACL** | 다국어 검색 | 다국어 검색 성능 평가 |
| **MKQA** | 다국어 QA·지식 | 다국어 질의응답 및 지식 베이스 |
| **MTEB** | 임베딩 종합 평가 | 임베딩 모델 전반적 성능 측정 |
| **KLUE / KorQuAD** | 한국어 NLU·QA | 한국어 자연어 이해 및 질의응답 |

### 단계 2: 출력 정제 모듈 (NR Module) 구현

인코더의 최종 임베딩 $Z \in \mathbb{R}^{T \times D}$ 에서 발생하는 **'세만틱 풀(Semantic-Pull)'** 효과를 수치화하여 노이즈를 제거한다.

1. **엔트로피 계산** — 각 토큰의 예측 확률 분포에서 $H(i) = -\sum p(i) \log p(i)$ 산출
2. **PDM 생성** — 해당 토큰 주변의 문맥적 신뢰도를 결합하여 확률 밀도 맵(Probability Density Map) 생성
3. **노이즈 분류** — Decision Tree 또는 경량 MLP를 통해 예측을 **'강한 예측(Strong)'** / **'약한 예측(Weak)'** 으로 분류하여 노이즈 제거

### 단계 3: 용량 제약을 위한 적응형 양자화 (SigmaQuant)

레이어별 민감도를 분석하여 비트 폭을 차등 할당함으로써 정확도 손실 없이 모델 크기를 최소화한다.

| 단계 | 기법 | 세부 내용 |
|------|------|-----------|
| **민감도 분석** | KL 발산 ($D_{KL}$) | 각 레이어의 양자화 민감도를 측정하여 비트 할당 우선순위 결정 |
| **비트 할당 (하위)** | 2~4 bit | 정확도에 영향이 적은 하위 레이어에 낮은 비트 폭 할당 |
| **비트 할당 (핵심)** | 8 bit | 핵심 특징 추출 레이어에 8비트 할당으로 품질 보존 |

---

## 4. 베이스라인 구성

실험 결과의 재현성과 신뢰성을 확보하기 위해 아래의 베이스라인 조합을 포함한다.

| 유형 | 구현체 | 특징 |
|------|--------|------|
| **Lexical Baseline** | BM25 계열 | 정확한 수식/구현은 선택지 다양, 세부 미지정 |
| **Dense Retrieval** | DPR류 bi-encoder | 밀집 벡터 기반 의미 검색 |
| **RAG Baseline** | 검색기 + 생성기 기본형 | 기본적인 RAG 파이프라인 |
| **Reranking** | Cross-encoder (SBERT / ColBERT) | 검색 결과 재랭킹 기반 정밀도 향상 |
| **Output Validation** | Guardrails / Outlines / llguidance | 스키마 검증 + 출력 포맷 강제 |

---

## 5. 평가 지표 (Evaluation Metrics)

성능과 자원 효율성을 동시에 평가하기 위해 **파레토 프런티어(Pareto Frontier)** 분석을 수행한다.

### 5.1 정밀도 성능

- **mIoU** — 이미지 분할 정확도 측정
- **F1-Score** — 텍스트 분류 균형 정확도
- **Recall@k, nDCG@k, MRR** — 검색 품질 (데이터셋 표준에 따름)
- **가양성 감소율 (FP Reduction Rate)** — NR Module 성능 지표
- **RAGAs 메트릭** — faithfulness / answer relevance / context precision·recall
- **Valid JSON Rate** — 스키마 준수율, 정책 위반율, 재시도율

### 5.2 추론 효율성

- 배치-1 추론 지연 시간 (Latency, ms)
- 초당 프레임 수 (FPS)
- p95 / p99 지연, QPS, 평균 토큰 수

### 5.3 자원 점유

- 모델 파일 크기 (MB)
- 피크 메모리 사용량 (Peak RAM / VRAM, GB)
- 연산량 (FLOPs)
- 인덱스 크기 (GB), 에너지 소비량

---

## 6. 재현성 체크리스트

- [ ] **데이터 버전 고정** — 문서 스냅샷, 청크 규칙, 전처리 규칙 고정
- [ ] **임베딩 모델 기록** — 모델 버전 / 차원 / 정규화 방식 문서화 (MTEB류 평가도 모델 특성에 민감)
- [ ] **인덱스 설정 기록** — HNSW / IVF / PQ 파라미터, ef_search 등 명시
- [ ] **재랭커 설정** — top-k → top-n, 배치 크기, 최대 입력 길이
- [ ] **출력 검증 정책** — 스키마, validator 버전, 실패 시 재시도 정책
- [ ] **자동 평가 프레임워크** — ragas / DeepEval 버전 고정 및 평가 호출 비용 모니터링

---

## 7. 기대 결과 (Expected Outcomes)

| | 목표 | 기대 수치 |
|-|------|-----------|
| 🎯 | **출력 정제율 향상** | NR 모듈을 통해 임계값 필터링으로 걸러지지 않던 미세 노이즈 제거 & **가양성 50% 이상 감소** 목표 |
| 💾 | **용량·비용 절감** | SigmaQuant 적응형 양자화로 고정 8-bit 모델 대비 정확도 손실 없이 **메모리 최대 40% 절감** & 에너지 소비 약 **20.6% 감소** |
| ⚡ | **서비스 지연 안정화** | 인코더-디코더 분리 구조를 통해 First-token Latency를 약 **47% 단축**하여 온디바이스 환경에서의 반응 속도 향상 |

---

> 본 실험 설계를 통해 인코더 활용 대처방안은
> **(1) 출력 정제율 상승 · (2) 용량·비용 절감 · (3) 서비스 지연 안정화**
> 의 세 목표를 정량적으로 검증함

---

## 8. 참고 문헌 및 레퍼런스

### 핵심 논문·기술보고서

| 제목 | 저자/기관 | 연도 | 요약 | 링크 |
|------|-----------|------|------|------|
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Jacob Devlin et al. (Google) | 2018 | Transformer 인코더 사전학습 기반 언어 표현 학습의 표준화를 촉진 | [arxiv](https://arxiv.org/abs/1810.04805) |
| Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Nils Reimers & Iryna Gurevych | 2019 | 문장 임베딩을 위한 시암 구조(SBERT) 제안 | [ACL](https://aclanthology.org/D19-1410/) |
| SimCSE: Simple Contrastive Learning of Sentence Embeddings | Tianyu Gao et al. | 2021 | 드롭아웃을 노이즈로 사용하는 단순 대조학습 기반 문장 임베딩 | [arxiv](https://arxiv.org/abs/2104.08821) |
| Dense Passage Retrieval for Open-Domain QA | Vladimir Karpukhin et al. (Meta 계열 연구) | 2020 | 질의/패시지 각각 인코딩→인덱스 구축→top-k 검색 구조를 체계화 | [ACL](https://aclanthology.org/2020.emnlp-main.550/) |
| ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT | Omar Khattab & Matei Zaharia | 2020 | 다중 벡터/late interaction 기반 고정밀 검색·재랭킹 계열 | [arxiv](https://arxiv.org/abs/2004.12832) |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Patrick Lewis et al. | 2020 | 비모수 메모리(검색) + 생성 모델 결합(RAG) 정식화 | [arxiv](https://arxiv.org/abs/2005.11401) |
| The Faiss library | Matthijs Douze et al. | 2024 | 벡터 DB 핵심 기능(검색·클러스터·압축·변환)으로서 FAISS 설계 원리 정리 | [arxiv](https://arxiv.org/abs/2401.08281) |
| Product Quantization for Nearest Neighbor Search | Hervé Jégou et al. | 2011 | 부분공간 분해 기반 PQ로 벡터를 짧은 코드로 근사 저장 | [ACM](https://dl.acm.org/doi/10.1109/TPAMI.2010.57) |
| Efficient and robust approximate nearest neighbor search using HNSW | Yury Malkov & Dmitry Yashunin | 2016/2018 | 계층적 small-world 그래프 기반 ANN(HNSW) 제안 | [arxiv](https://arxiv.org/abs/1603.09320) |
| Announcing ScaNN: Efficient Vector Similarity Search | Google Research | 2020 | 검색 공간 pruning+양자화 기반 벡터 검색 라이브러리 공개 소개 | [blog](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/) |
| Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference | Benoit Jacob et al. | 2018 | 정수 산술 기반 추론을 위한 양자화 스킴과 학습 방법 | [arxiv](https://arxiv.org/abs/1712.05877) |
| GPTQ: Accurate Post-Training Quantization for GPTs | Elias Frantar et al. | 2022 | 3–4bit PTQ로 대형 GPT 계열 메모리/속도 개선을 목표 | [arxiv](https://arxiv.org/abs/2210.17323) |
| AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | Ji Lin et al. | 2023/2024 | 활성값 기반으로 중요한 채널을 보호해 양자화 오차를 줄임 | [arxiv](https://arxiv.org/abs/2306.00978) |
| Distilling the Knowledge in a Neural Network | Geoffrey Hinton et al. | 2015 | 지식 증류(KD) 기본 패러다임 제시 | [arxiv](https://arxiv.org/abs/1503.02531) |
| DistilBERT, a distilled version of BERT | Victor Sanh et al. (Hugging Face) | 2019 | BERT를 작은 모델로 증류해 속도/비용 개선 | [arxiv](https://arxiv.org/abs/1910.01108) |
| TinyBERT: Distilling BERT for Natural Language Understanding | Xiaoqi Jiao et al. (Huawei) | 2019/2020 | Transformer 특화 지식 증류로 경량 모델 성능 확보 | [arxiv](https://arxiv.org/abs/1909.10351) |
| MiniLM: Deep Self-Attention Distillation | Wenhui Wang et al. | 2020 | 자기어텐션 지식을 증류해 효율적 압축 | [arxiv](https://arxiv.org/abs/2002.10957) |
| LLMLingua: Compressing Prompts for Accelerated Inference of LLMs | Huiqiang Jiang et al. | 2023 | 토큰 단위 압축으로 긴 프롬프트 비용 절감, 높은 압축비 실험 | [arxiv](https://arxiv.org/abs/2310.05736) |
| LLMLingua-2: Data Distillation for Prompt Compression | Zhuoshi Pan et al. (Microsoft) | 2024 | 데이터 증류 기반 범용 프롬프트 압축(ACL Findings) | [arxiv](https://arxiv.org/abs/2403.12968) |
| In-context Autoencoder (ICAE) for Context Compression | Tao Ge et al. (Microsoft) | 2023/2024 | 긴 컨텍스트를 메모리 슬롯으로 압축해 효율 추론 | [arxiv](https://arxiv.org/abs/2307.06945) |
| Perceiver IO: A General Architecture for Structured Inputs & Outputs | Andrew Jaegle et al. | 2021 | 잠재 공간(latent space)에서 대규모 입력 처리 구조 | [pdf](https://www.drewjaegle.com/pdfs/jaegle2021perceiver_io.pdf) |
| CLIP: Learning Transferable Visual Models From Natural Language Supervision | Alec Radford et al. (OpenAI) | 2021 | 이미지-텍스트 공동 임베딩 듀얼 인코더로 제로샷 전이 | [arxiv](https://arxiv.org/abs/2103.00020) |
| SigLIP: Sigmoid Loss for Language Image Pre-Training | Xiaohua Zhai et al. | 2023 | CLIP류 대비 손실 함수 변경으로 스케일링/성능 개선 시도 | [arxiv](https://arxiv.org/abs/2303.15343) |
| ALIGN: Scaling Up Visual and Vision-Language Representation Learning | Chao Jia et al. | 2021 | noisy image-text에서 dual-encoder로 스케일링 학습 | [arxiv](https://arxiv.org/abs/2102.05918) |
| KLUE: Korean Language Understanding Evaluation | Sungjoon Park et al. | 2021 | 한국어 NLU 8개 과업 벤치마크·평가 레시피 제시 | [arxiv](https://arxiv.org/abs/2105.09680) |
| KorQuAD1.0 | Seungyoung Lim et al. | 2019 | 한국어 기계독해 QA 데이터셋 공개 | [arxiv](https://arxiv.org/abs/1909.07005) |

---

### 대표 오픈소스·공식 프로젝트

#### ANN / 벡터 검색

| 프로젝트 | 범주 | 요약 | 링크 |
|----------|------|------|------|
| **FAISS** | ANN/벡터 검색 라이브러리 | 대규모 벡터 유사도 검색·클러스터·압축 지원 | [GitHub](https://github.com/facebookresearch/faiss) |
| **FAISS Wiki** | 공식 문서 | 인덱스 선택 가이드, 벡터 코덱(SQ/PQ) 등 | [Wiki](https://github.com/facebookresearch/faiss/wiki) |
| **ScaNN** | ANN 라이브러리 | pruning+quantization 기반 벡터 검색 구현 | [GitHub](https://github.com/google-research/google-research/tree/master/scann) |
| **hnswlib** | ANN 라이브러리 | HNSW 구현(C++/Python) | [GitHub](https://github.com/nmslib/hnswlib) |
| **Annoy** | ANN 라이브러리 | 파일 기반 read-only 인덱스(메모리 매핑) | [GitHub](https://github.com/spotify/annoy) |

#### 벡터 DB

| 프로젝트 | 범주 | 요약 | 링크 |
|----------|------|------|------|
| **Milvus** | 벡터 DB | 대규모 벡터 DB(스케일 지향) | [GitHub](https://github.com/milvus-io/milvus) |
| **Qdrant** | 벡터 DB | 벡터 + payload 필터링 중심 엔진 | [GitHub](https://github.com/qdrant/qdrant) |
| **Weaviate** | 벡터 DB | 오브젝트+벡터, 하이브리드/쿼리 기능 | [GitHub](https://github.com/weaviate/weaviate) |
| **pgvector** | DB 확장 | Postgres 벡터 검색(approx/exact) | [GitHub](https://github.com/pgvector/pgvector) |

#### 임베딩 / 재랭킹

| 프로젝트 | 범주 | 요약 | 링크 |
|----------|------|------|------|
| **sentence-transformers** | 임베딩/재랭킹 프레임워크 | 문장 임베딩·cross-encoder reranker 지원 | [GitHub](https://github.com/UKPLab/sentence-transformers) |
| **FlagEmbedding (BGE)** | 임베딩 프레임워크 | BGE-M3 등 다중 검색 모드 지원 | [GitHub](https://github.com/FlagOpen/FlagEmbedding) |

#### RAG 평가 / 출력 검증

| 프로젝트 | 범주 | 요약 | 링크 |
|----------|------|------|------|
| **ragas** | RAG 평가 | RAGAs 기반 평가/메트릭 구현·문서 | [GitHub](https://github.com/vibrantlabsai/ragas) |
| **DeepEval** | LLM 평가 | 테스트 프레임워크 + G-Eval 등 메트릭 | [GitHub](https://github.com/confident-ai/deepeval) |
| **Guardrails** | 출력 검증/정제 | validator 기반 입력·출력 품질 제어 | [GitHub](https://github.com/guardrails-ai/guardrails) |
| **Outlines** | 구조화 출력 | JSON 스키마/문법 기반 구조화 생성 | [GitHub](https://github.com/dottxt-ai/outlines) |
| **llguidance** | constrained decoding | 고속 문법 제약 디코딩 라이브러리 | [GitHub](https://github.com/guidance-ai/llguidance) |

#### 양자화

| 프로젝트 | 범주 | 요약 | 링크 |
|----------|------|------|------|
| **bitsandbytes** | 양자화 구현 | 8bit/4bit 연산·옵티마이저 제공 | [GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| **GPTQ 구현체** | 양자화 구현 | GPTQ 논문 코드 | [GitHub](https://github.com/IST-DASLab/gptq) |
| **AWQ 구현체** | 양자화 구현 | AWQ 코드 | [GitHub](https://github.com/mit-han-lab/llm-awq) |

#### 추론 최적화 / 서빙

| 프로젝트 | 범주 | 요약 | 링크 |
|----------|------|------|------|
| **TensorRT-LLM** | 추론 최적화 | LLM 서빙 최적화(양자화/배치/KV캐시 등) | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **Triton Inference Server** | 모델 서빙 | 다양한 프레임워크 모델 서빙 | [GitHub](https://github.com/triton-inference-server/server) |
| **vLLM** | LLM 서빙 | 고처리량·메모리 효율 LLM 서빙 | [GitHub](https://github.com/vllm-project/vllm) |
| **llama.cpp** | 로컬 추론 | 로컬/온디바이스 추론(포맷: GGUF) | [GitHub](https://github.com/ggml-org/llama.cpp) |

#### 한국어 모델

| 프로젝트 | 범주 | 요약 | 링크 |
|----------|------|------|------|
| **KoBERT** | 한국어 인코더 | SK Telecom 주도 한국어 BERT | [GitHub](https://github.com/SKTBrain/KoBERT) |
| **KoELECTRA** | 한국어 인코더 | 한국어 ELECTRA 사전학습 모델 배포 | [GitHub](https://github.com/monologg/KoELECTRA) |
| **KoCLIP** | 한국어 멀티모달 인코더 | 한국어 CLIP 포트/학습 시도 | [GitHub](https://github.com/jaketae/koclip) |
