"""
AudioSeal MP3 공격 + 복호 + BER/DSR/STOI 측정 파이프라인
=========================================================
입력: wm_wav/ (AudioSeal 삽입된 wav)
출력: results_mp3/ (CSV 결과)
"""

import os
import torch
import numpy as np
import soundfile as sf
import subprocess
import tempfile
import csv
from pathlib import Path
from pystoi import stoi

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

from audioseal import AudioSeal

# ============================================================
#  설정
# ============================================================
WM_WAV_DIR  = Path(r"C:\Users\user\OneDrive\Desktop\audioseal\wm_wav")
OUTPUT_DIR  = Path(r"C:\Users\user\OneDrive\Desktop\audioseal\results_mp3")
SAMPLE_RATE = 16000
BITRATES    = [128, 96, 64, 48, 32]  # kbps

# 삽입 시 사용한 고정 메시지 (embed_audioseal.py와 동일)
FIXED_MESSAGE = torch.tensor(
    [[1,0,1,1,0,0,1,0,1,1,0,1,0,1,0,0]], dtype=torch.float32
)  # shape: (1, 16)

DSR_THRESHOLD = 0.95  # 정책 기준 95%

# ============================================================
#  MP3 공격 함수
# ============================================================
def mp3_attack(audio: np.ndarray, sr: int, bitrate: int) -> np.ndarray:
    """WAV → MP3 압축 → WAV 재변환"""
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path  = Path(tmpdir) / "input.wav"
        mp3_path = Path(tmpdir) / "compressed.mp3"
        out_path = Path(tmpdir) / "output.wav"

        sf.write(str(in_path), audio, sr)

        # WAV → MP3
        subprocess.run([
            "ffmpeg", "-y", "-i", str(in_path),
            "-b:a", f"{bitrate}k",
            str(mp3_path)
        ], capture_output=True)

        # MP3 → WAV
        subprocess.run([
            "ffmpeg", "-y", "-i", str(mp3_path),
            str(out_path)
        ], capture_output=True)

        attacked, _ = sf.read(str(out_path), always_2d=False)
        attacked = attacked.astype(np.float32)
        attacked = np.clip(attacked, -1.0, 1.0)

        # 길이 맞추기
        min_len = min(len(audio), len(attacked))
        return attacked[:min_len], audio[:min_len]

# ============================================================
#  BER 계산
# ============================================================
def calculate_ber(decoded_bits: torch.Tensor, original: torch.Tensor) -> float:
    """BER = 오류 비트 수 / 전체 비트 수"""
    decoded = (decoded_bits > 0).float()
    errors  = (decoded != original).float().sum().item()
    return errors / original.numel()

# ============================================================
#  DSR 계산 (샘플 단위)
# ============================================================
def calculate_dsr(decoded_bits: torch.Tensor, original: torch.Tensor) -> int:
    """16비트 전부 일치하면 1, 아니면 0"""
    decoded = (decoded_bits > 0).float()
    return 1 if torch.equal(decoded, original) else 0

# ============================================================
#  메인
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    print("AudioSeal 모델 로딩 중...")
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    detector.eval()
    print("완료!\n")

    # ffmpeg 확인
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if result.returncode != 0:
        print("❌ ffmpeg가 설치되어 있지 않습니다!")
        return

    # wav 파일 수집
    wav_files = sorted(WM_WAV_DIR.rglob("*.wav"))
    print(f"총 파일 수: {len(wav_files)}개")
    print(f"비트레이트: {BITRATES} kbps")
    print("=" * 60)

    # 비트레이트별 결과 저장
    summary = {}

    for bitrate in BITRATES:
        print(f"\n[MP3 {bitrate}kbps] 처리 시작...")

        ber_list  = []
        dsr_list  = []
        stoi_list = []
        error_count = 0

        rows = []

        for i, p in enumerate(wav_files):
            try:
                # 워터마킹된 오디오 로드
                audio, sr = sf.read(str(p), always_2d=False)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.float32)
                audio = np.clip(audio, -1.0, 1.0)

                if sr != SAMPLE_RATE:
                    error_count += 1
                    continue

                # MP3 공격
                attacked, original_trimmed = mp3_attack(audio, sr, bitrate)

                # STOI 계산 (공격 전후 비교)
                stoi_score = stoi(original_trimmed, attacked, sr, extended=False)

                # tensor 변환
                attacked_tensor = torch.tensor(attacked).unsqueeze(0).unsqueeze(0)

                # 복호
                with torch.no_grad():
                    _, decoded_bits = detector(attacked_tensor, sample_rate=SAMPLE_RATE)

                # 샘플별 평균 (시간축 평균)
                decoded_mean = decoded_bits.mean(dim=2)  # (1, 16)

                # BER / DSR
                ber = calculate_ber(decoded_mean, FIXED_MESSAGE)
                dsr = calculate_dsr(decoded_mean, FIXED_MESSAGE)

                ber_list.append(ber)
                dsr_list.append(dsr)
                stoi_list.append(stoi_score)

                rows.append({
                    "file"      : p.name,
                    "bitrate"   : bitrate,
                    "ber"       : round(ber, 4),
                    "dsr"       : dsr,
                    "stoi"      : round(stoi_score, 4),
                })

                if (i + 1) % 50 == 0 or (i + 1) == len(wav_files):
                    print(f"  [{i+1}/{len(wav_files)}] 완료")

            except Exception as e:
                print(f"  💥 오류: {p.name} → {e}")
                error_count += 1

        # CSV 저장
        csv_path = OUTPUT_DIR / f"mp3_{bitrate}kbps.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file","bitrate","ber","dsr","stoi"])
            writer.writeheader()
            writer.writerows(rows)

        # 요약
        avg_ber  = np.mean(ber_list) if ber_list else 1.0
        dsr_rate = np.mean(dsr_list) if dsr_list else 0.0
        avg_stoi = np.mean(stoi_list) if stoi_list else 0.0

        status = "✅" if dsr_rate >= DSR_THRESHOLD else "🔴 붕괴"
        print(f"  → BER: {avg_ber:.4f} | DSR: {dsr_rate*100:.1f}% {status} | STOI: {avg_stoi:.4f}")

        summary[bitrate] = {
            "avg_ber" : avg_ber,
            "dsr_rate": dsr_rate,
            "avg_stoi": avg_stoi,
            "errors"  : error_count
        }

    # 최종 요약
    print("\n" + "=" * 60)
    print("  MP3 공격 실험 완료")
    print("=" * 60)
    print(f"  {'비트레이트':<12} {'BER':<10} {'DSR':<12} {'STOI':<10} {'판정'}")
    print(f"  {'-'*55}")
    for br, s in summary.items():
        status = "✅" if s['dsr_rate'] >= DSR_THRESHOLD else "🔴 붕괴"
        print(f"  {br}kbps{'':<8} {s['avg_ber']:.4f}{'':<6} "
              f"{s['dsr_rate']*100:.1f}%{'':<8} {s['avg_stoi']:.4f}{'':<6} {status}")
    print("=" * 60)

if __name__ == "__main__":
    np.random.seed(42)
    main()