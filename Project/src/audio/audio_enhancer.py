import os
import torch
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt

class AudioEnhancer:
    """
    [STT 최적화 버전]
    전화망 음성(8kHz 대역)을 16kHz Whisper 모델에 맞게 업샘플링 및 정제하는 전처리 클래스
    """
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[AudioEnhancer] Loading Silero VAD on {self.device}...")

        # Silero VAD 로드
        self.vad_model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", onnx=False, trust_repo=True
        )
        (self.get_speech_timestamps, _, _, _, _) = utils
        self.vad_model.to(self.device)

    # [수정됨/확인] 전화망 대역폭 필터링 (Standard PSTN Bandwidth)
    def bandpass_filter(self, audio):
        """
        사람 목소리의 주요 포먼트(Formant)가 존재하는 300~3400Hz만 남기고
        저주파 웅웅거림(Hum)과 고주파 히스(Hiss) 노이즈를 제거합니다.
        """
        # 차단 주파수 설정
        lowcut = 300.0
        highcut = 3400.0 
        
        # [안정성 추가] 오디오 길이가 너무 짧으면 필터링 시 에러 발생 가능하므로 예외 처리
        if len(audio) <= self.target_sr * 0.01: # 10ms 미만
            return audio

        # Butterworth Bandpass Filter (Order 10: 가파르게 깎음)
        sos = butter(10, [lowcut, highcut], btype='band', fs=self.target_sr, output='sos')
        filtered = sosfiltfilt(sos, audio)
        
        # 필터링 과정에서 위상이 틀어지거나 값이 튀는 것을 방지하기 위해 float32 변환
        return filtered.astype(np.float32)

    # [수정됨/확인] 노이즈 감소 (Spectral Gating)
    def reduce_noise(self, audio):
        """
        배경 노이즈를 제거하되, 목소리 왜곡(Artifact)을 최소화하기 위해 강도를 0.75로 제한합니다.
        """
        try:
            # 오디오가 너무 짧으면 NR 건너뜀
            if len(audio) < self.target_sr * 0.1:
                return audio

            reduced_audio = nr.reduce_noise(
                y=audio, 
                sr=self.target_sr, 
                prop_decrease=0.75,  # [설정값 유지] 1.0은 목소리 손상 위험, 0.75가 안전
                n_fft=2048,          # 주파수 해상도 확보
                stationary=True,     # [가정] 통화 배경 소음은 일정하다고 가정 (팬 소음 등)
                use_tqdm=False,
                n_jobs=1             # 병렬 처리 오버헤드 방지
            )
            return reduced_audio
        except Exception as e:
            print(f"⚠️ Noise reduction skipped due to error: {e}")
            return audio

    # [수정됨/확인] VAD 기반 무음 제거 및 Padding 적용
    def vad_trim(self, audio):
        """
        순수 발화 구간만 추출하되, 앞뒤에 여유분(Padding)을 두어 
        Whisper 모델이 문맥을 파악할 수 있도록 합니다.
        """
        wav_torch = torch.from_numpy(audio).float().to(self.device)
        
        # [안정성 추가] 차원 문제 방지 (1D Array -> 2D Tensor)
        if wav_torch.ndim == 1:
            wav_torch = wav_torch.unsqueeze(0)

        # [파라미터 설명]
        # threshold=0.4: 대출 사기형 데이터 중 목소리가 작은 경우를 고려해 민감도 높임 (기본 0.5)
        # min_speech_duration_ms=250: '네', '아니오' 같은 짧은 대답 보존
        # min_silence_duration_ms=500: 문장 중간의 짧은 쉼은 자르지 않음 (문맥 보존)
        speech_timestamps = self.get_speech_timestamps(
            wav_torch, 
            self.vad_model, 
            threshold=0.4, 
            min_speech_duration_ms=250, 
            min_silence_duration_ms=500,
            sampling_rate=self.target_sr
        )

        if not speech_timestamps:
            # 발화가 감지되지 않으면, 원본이 너무 시끄러운 경우일 수 있으므로 
            # 무음 처리하거나 원본 반환 (여기서는 원본 반환 선택)
            return audio

        # [핵심 수정] Padding 300ms 적용
        # Whisper는 문장의 시작과 끝의 천이 구간(Transient) 정보가 필요함
        pad_samples = int(self.target_sr * 0.3) 
        
        segments = []
        for ts in speech_timestamps:
            start = max(0, ts["start"] - pad_samples)
            end = min(len(audio), ts["end"] + pad_samples)
            
            # 유효한 구간인 경우만 추가
            if end > start:
                segments.append(audio[start:end])

        if not segments:
            return audio
            
        # 추출된 구간들을 하나로 이어 붙임
        return np.concatenate(segments).astype(np.float32)

    def normalize(self, audio, target_rms=0.1):
        """
        모든 오디오의 볼륨(에너지)을 일정 수준으로 맞춤
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6: # 너무 조용한 경우 (Divide by zero 방지)
            return audio
            
        normalized = audio * (target_rms / rms)
        
        # 증폭 후 -1.0 ~ 1.0 범위를 벗어나는 피크(Clipping) 제어
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def enhance(self, input_path, output_path):
        try:
            # [로드] librosa는 기본적으로 float32로 로드하며, sr=16000으로 리샘플링 수행
            audio, _ = librosa.load(input_path, sr=self.target_sr, mono=True)
            
            if len(audio) == 0: 
                return False

            # [파이프라인 실행]
            # 1. Bandpass: 불필요한 주파수 제거 (가장 먼저 수행하여 NR 부하 감소)
            audio = self.bandpass_filter(audio)
            
            # 2. Noise Reduction: 남은 대역 내의 잡음 제거
            audio = self.reduce_noise(audio)
            
            # 3. VAD: 무음 제거 (STT 처리 속도 향상 및 환각 방지)
            audio = self.vad_trim(audio)
            
            # 4. Normalize: 볼륨 평준화
            audio = self.normalize(audio)

            # [저장]
            sf.write(output_path, audio, self.target_sr)
            return True
            
        except Exception as e:
            print(f"❌ Enhancement Error processing {input_path}: {e}")
            return False