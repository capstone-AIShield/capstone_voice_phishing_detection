import os
import re
import tempfile
import torch
# [라이브러리 확인] faster_whisper가 설치되어 있어야 합니다.
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from audio_enhancer import AudioEnhancer

class AudioProcessor:
    """
    [Strict Mode]
    Enhancement -> Whisper (High Accuracy) -> Filtering -> T5 Correction -> Final Filtering
    """
    def __init__(self, whisper_model_size="deepdml/faster-whisper-large-v3-turbo-ct2", t5_model_name="j5ng/et5-typos-corrector"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Whisper는 int8 양자화로 메모리 절약 (성능 저하 미미함)
        compute_type = "float16" if self.device == "cuda" else "int8"

        self.enhancer = AudioEnhancer()

        print(f"[AudioProcessor] Loading Whisper '{whisper_model_size}'...")
        self.whisper = WhisperModel(whisper_model_size, device=self.device, compute_type=compute_type)

        print(f"[AudioProcessor] Loading T5 Text Correction Model '{t5_model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name, use_fast=False)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name).to(self.device)
        
        # [핵심] 발견되면 문장 전체를 즉시 삭제할 블랙리스트
        self.blacklist = [
            # 1. Whisper 환각/특수 토큰
            "correct", "hate-speech", "generated", "silence", "subtitles", "transcript", 
            "MBC", "SBS", "KBS", "YTN", "뉴스", # 뉴스 멘트 제거
            # 2. 방송/자막 크레딧
            "자막", "배달의 민족", "저작권", "시청해 주셔서", "구독과 좋아요", "알림 설정", 
            "Provided by", "Auto-generated",
            # 3. 무의미한 문장 끝맺음/오인식
            "공인이 진행 의사", "안내 좀 뭐 및", "오디오 자막", "허허허"
        ]

    def clean_text_basic(self, text: str) -> str:
        """기본적인 텍스트 정제 (특수문자 제거 등)"""
        # 한글, 영문, 숫자, 기본 구두점만 남김
        text = re.sub(r'[^0-9a-zA-Z가-힣\s.,?!]', '', text)

        # [수정됨: 문장 부호 우선순위 처리]
        # 학술적 정의: Punctuation Normalization (구두점 정규화)
        # 현상: '여보세요?.' 처럼 물음표/느낌표 뒤에 마침표가 붙는 경우
        # 해결: 의미가 강한 ? 나 ! 를 남기고 . 을 제거함
        
        # 패턴 1: 물음표나 느낌표 뒤에 마침표가 오는 경우 (?. -> ?)
        text = re.sub(r'([?!])\s*\.+', r'\1', text)
        
        # 패턴 2: 마침표 뒤에 물음표나 느낌표가 오는 경우 (.? -> ?)
        text = re.sub(r'\.+\s*([?!])', r'\1', text)

        # [기존 로직 유지] 동일한 부호 반복 정리 (.. -> ., !! -> !)
        text = re.sub(r'([?!.])\1+', r'\1', text)
        
        # 다중 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def is_valid_sentence(self, text: str) -> bool:
        """
        문장이 데이터셋으로 쓸 만한지 엄격하게 검사합니다.
        False 반환 시 해당 문장은 데이터셋에서 제외됩니다.
        """
        if len(text) < 2: return False
        
        # [삭제 로직 1] 블랙리스트 키워드 포함 여부
        text_lower = text.lower()
        for keyword in self.blacklist:
            if keyword.lower() in text_lower:
                return False
        
        # [삭제 로직 2] 의미 없는 반복 (예: "네네네네네")
        no_space = text.replace(' ', '')
        if len(no_space) > 10:
            # 유니크한 글자 수가 전체 길이의 10% 미만이면 반복으로 간주
            if len(set(no_space)) / len(no_space) < 0.1: return False
        
        # [삭제 로직 3] 단어 반복 (예: "진짜 진짜 진짜")
        if re.search(r'(\b\w+\b)( \1){3,}', text): return False
        
        # [삭제 로직 4] 문장이 불완전하게 끝나는 경우 (Whisper 오류 패턴)
        if text.strip().endswith((' 및.', ' 또는.', ' 뭐.', ' 등.', ' 및')): return False
        
        return True

    def remove_duplicates(self, sentences: list) -> list:
        """연속된 중복 문장 제거"""
        unique = []
        prev = None
        for s in sentences:
            s = s.strip()
            if not s: continue
            if s != prev: unique.append(s)
            prev = s
        return unique

    def t5_correction(self, sentences: list, max_length=256, batch_size=16) -> list:
        """
        T5 모델을 사용하여 오탈자 교정 수행
        """
        valid_indices = []
        valid_inputs = []
        final_results = list(sentences) # 원본 복사

        # T5 입력 전처리
        for idx, sent in enumerate(sentences):
            if self.is_valid_sentence(sent):
                valid_indices.append(idx)
                # T5 모델에 따라 prefix가 필요할 수 있음 (일반적으로 "correct: " 사용)
                valid_inputs.append("correct: " + sent)

        if not valid_inputs:
            return sentences

        # 배치 처리 (OOM 방지)
        for i in range(0, len(valid_inputs), batch_size):
            batch_texts = valid_inputs[i : i + batch_size]
            batch_indices = valid_indices[i : i + batch_size]

            try:
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.t5_model.generate(
                        **inputs, 
                        max_length=max_length,
                        num_beams=2, # 교정 품질을 위해 beam search 살짝 적용
                        early_stopping=True
                    )
                
                decoded_batch = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for original_idx, corrected in zip(batch_indices, decoded_batch):
                    # [수정됨: T5 결과물에 대해서도 중복 부호 정규화 재적용]
                    # T5가 가끔 '네?.' 형태로 교정 결과를 내놓을 수 있음
                    corrected = re.sub(r'([?!])\s*\.+', r'\1', corrected)
                    corrected = re.sub(r'\.+\s*([?!])', r'\1', corrected)
                    corrected = re.sub(r'([?!.])\1+', r'\1', corrected).strip()
                    
                    # 교정 결과가 오히려 이상해지는 경우(빈 문자열 등) 체크
                    if not corrected or not self.is_valid_sentence(corrected):
                        # 교정 실패 시 원본 유지 혹은 삭제? -> 여기서는 빈 문자열로 삭제 처리
                        final_results[original_idx] = "" 
                    else:
                        final_results[original_idx] = corrected

            except Exception as e:
                print(f"⚠️ T5 Correction Error (Batch {i}): {e}")
                # 에러 발생 시 해당 배치는 원본 유지

        return [s for s in final_results if s.strip()]

    def process_file(self, audio_path: str, t5_correct=False) -> list:
        """
        단일 오디오 파일 처리 파이프라인:
        Enhance -> Whisper -> Cleaning -> KSS Split -> Filtering -> (T5) -> Result
        """
        if not os.path.exists(audio_path):
            return []

        # 임시 파일 생성 (Enhanced Audio)
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        enhanced_path = tmp_file.name
        tmp_file.close()

        try:
            # 1. 오디오 전처리 (VAD, NR, Filter)
            success = self.enhancer.enhance(audio_path, enhanced_path)
            target_path = enhanced_path if success else audio_path
            
            # 2. Whisper STT (파라미터 최적화됨)
            segments, info = self.whisper.transcribe(
                target_path,
                beam_size=5,                # [확인] 정확도를 위해 5로 설정
                language="ko",
                condition_on_previous_text=False, # 문맥 의존성 제거 (환각 방지)
                repetition_penalty=1.2,     # [확인] 반복 생성을 억제
                vad_filter=False,           # 전처리 단계에서 이미 VAD 수행함
                temperature=0.0,            # 결정론적 결과
                no_speech_threshold=0.6,    # 묵음 확률 임계값
                # log_prob_threshold=-1.0   # (선택) 확신 없는 문장은 버리기
            )
            
            full_text = " ".join([seg.text for seg in segments])

        except Exception as e:
            print(f"❌ STT Error {audio_path}: {e}")
            return []
            
        finally:
            # 임시 파일 정리
            if os.path.exists(enhanced_path):
                try:
                    os.remove(enhanced_path)
                except:
                    pass

        # 3. 텍스트 후처리
        cleaned_text = self.clean_text_basic(full_text)

        # 문장 분리 (KSS 권장, 없으면 split)
        try:
            import kss
            sentences = kss.split_sentences(cleaned_text)
        except ImportError:
            # [수정됨: 문장 분리 fallback 로직 개선]
            # 기존: replace("?", "?.").replace("!", "!.") -> 이 코드가 '여보세요?.'를 만드는 원인 중 하나일 수 있음
            # 개선: 정규식을 이용해 문장 분리만 수행하고 불필요한 마침표 추가를 막음
            sentences = re.split(r'(?<=[.?!])\s+', cleaned_text)

        # 1차 필터링
        sentences = [s for s in sentences if self.is_valid_sentence(s)]

        # 4. T5 문법 교정 (옵션)
        if t5_correct:
            sentences = self.t5_correction(sentences)
            # 2차 필터링 (교정 후 검증)
            sentences = [s for s in sentences if self.is_valid_sentence(s)]

        # 중복 제거 및 최종 길이 필터링
        sentences = self.remove_duplicates(sentences)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 3]

        return sentences