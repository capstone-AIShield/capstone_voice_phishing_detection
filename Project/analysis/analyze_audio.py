import os
import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Silero VAD 모델 로드 (기존과 동일)
device = "cuda" if torch.cuda.is_available() else "cpu"
vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False, trust_repo=True)
(get_speech_timestamps, _, _, _, _) = utils
vad_model.to(device)

# [수정됨] 분석 대상 폴더 리스트를 인자로 추가
def analyze_audio_folder_recursive(root_folder, target_sr=16000, target_subdirs=None):
    stats = []
    
    # [추가됨] 필터링할 폴더 이름들이 지정되지 않았을 경우를 대비한 기본값 처리
    if target_subdirs is None:
        target_subdirs = []

    for dirpath, _, filenames in os.walk(root_folder):
        
        # [추가됨] 현재 경로(dirpath)가 타겟 폴더 중 하나를 포함하는지 확인하는 로직
        # target_subdirs가 비어있지 않은 경우, 현재 경로에 해당 폴더명이 없다면 건너뜁니다(Skip).
        if target_subdirs:
            # 운영체제별 경로 구분자(\ 혹은 /) 문제를 피하기 위해 경로 정규화 후 확인
            norm_path = os.path.normpath(dirpath)
            if not any(target in norm_path for target in target_subdirs):
                continue

        for filename in tqdm(filenames, desc=f"Processing {dirpath}"):
            # [수정됨] 이전에 요청하신 대로 wav와 mp3만 처리하도록 필터링
            if not filename.lower().endswith((".wav", ".mp3")):
                continue
            
            file_path = os.path.join(dirpath, filename)
            relative_folder = os.path.relpath(dirpath, root_folder)
            
            try:
                # (이하 분석 로직은 기존과 동일합니다)
                audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
                duration = len(audio) / sr
                rms = np.sqrt(np.mean(audio ** 2))
                zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
                clipping_ratio = np.mean(np.abs(audio) >= 1.0)
                
                wav_torch = torch.from_numpy(audio).float().to(device)
                timestamps = get_speech_timestamps(wav_torch, vad_model, sampling_rate=sr, threshold=0.45)
                
                if timestamps:
                    speech_len = sum([ts['end']-ts['start'] for ts in timestamps]) / sr
                    num_segments = len(timestamps)
                    mean_segment_len = speech_len / num_segments
                else:
                    speech_len = 0
                    num_segments = 0
                    mean_segment_len = 0
                
                silence_len = duration - speech_len
                speech_ratio = speech_len / duration if duration > 0 else 0
                
                stats.append({
                    "folder": relative_folder,
                    "file": filename,
                    "duration": duration,
                    "rms": rms,
                    "zcr": zcr,
                    "clipping_ratio": clipping_ratio,
                    "speech_len": speech_len,
                    "silence_len": silence_len,
                    "speech_ratio": speech_ratio,
                    "num_segments": num_segments,
                    "mean_segment_len": mean_segment_len
                })
                
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
    
    return stats

# (summarize_stats_by_folder 함수는 기존과 동일하므로 생략)
# 필요하다면 이전 코드의 해당 함수를 그대로 사용하시면 됩니다.
def summarize_stats_by_folder(stats, save_folder="analysis_results"):
    os.makedirs(save_folder, exist_ok=True)
    df = pd.DataFrame(stats)
    
    if df.empty:
        print("⚠️ 분석된 데이터가 없습니다. 폴더 경로나 파일 확장자를 확인하세요.")
        return df, None, None

    summary = df.describe().T
    summary.to_csv(os.path.join(save_folder, "audio_summary_stats.csv"))
    df.to_csv(os.path.join(save_folder, "audio_file_stats.csv"), index=False)
    
    folder_summary = df.groupby("folder").agg({
        "duration": ["mean","std","min","max"],
        "rms": ["mean","std"],
        "zcr": ["mean","std"],
        "clipping_ratio": ["mean","std"],
        "speech_ratio": ["mean","std"],
        "mean_segment_len": ["mean","std"],
        "num_segments": ["mean","std"]
    })
    folder_summary.to_csv(os.path.join(save_folder, "folder_summary_stats.csv"))
    
    # 시각화 부분은 데이터가 충분할 때만 실행
    if len(df) > 1:
        df.hist(column=["duration", "rms", "zcr", "clipping_ratio", "speech_ratio", "mean_segment_len"], bins=30, figsize=(12,8))
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "histograms.png"))
        plt.close()
        
        df.boxplot(column=["duration", "rms", "speech_ratio"], figsize=(10,5))
        plt.savefig(os.path.join(save_folder, "boxplots.png"))
        plt.close()
    
    print(f"✅ 분석 결과가 '{save_folder}' 폴더에 저장되었습니다.")
    return df, summary, folder_summary

# -----------------------
# 사용 예시 (수정됨)
# -----------------------
root_folder = "../보이스 피싱 데이터(금감원)"

# [추가됨] 분석하고 싶은 특정 폴더 이름 리스트 정의
target_folders = ["대출 사기형", "바로 이 목소리", "수사기관 사칭형"]

# [수정됨] target_subdirs 인자에 리스트 전달
stats = analyze_audio_folder_recursive(root_folder, target_subdirs=target_folders)

if stats:
    df, summary, folder_summary = summarize_stats_by_folder(stats, save_folder="audio_analysis_results")
else:
    print("분석된 파일이 없습니다.")