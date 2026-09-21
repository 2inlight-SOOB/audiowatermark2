"""
AudioSeal 2단계 평가 — 워터마크 삽입 직후(무공격), 비교 기준: 원본
====================================================================
지표: BER, NC, DSR, SI-SNR, ViSQOL(선택), ODG(선택), 차이 스펙트로그램
PESQ/STOI는 음성 전용이라 제외.

대상 파일 매칭 규칙:
  original/A01_40/LA_D_xxxx.wav  ↔  embed/A01_40/wm_LA_D_xxxx.wav
  (as_embed_audioseal.py 에서 붙인 wm_ 접두사 기준)
"""

import os
import csv
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

from audioseal import AudioSeal

matplotlib.rcParams["font.family"] = "Malgun Gothic"   # 한글 깨짐 방지 (Windows)
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
#  설정
# ============================================================
ORIGINAL_DIR = Path(r"C:\Users\user\OneDrive\Desktop\new_watermark\original")
WM_DIR       = Path(r"C:\Users\user\OneDrive\Desktop\new_watermark\embed")
RESULT_DIR   = Path(r"C:\Users\user\OneDrive\Desktop\new_watermark\eval_step2_noattack")
SPEC_DIR     = RESULT_DIR / "diff_spectrograms"

SAMPLE_RATE = 16000
FIXED_MESSAGE = torch.tensor(
    [[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]], dtype=torch.float32
)

DETECTION_THRESHOLD = 0.5   # DSR 판정 임계값 (워터마크로 판정된 프레임 비율 기준)

SAVE_DIFF_SPECTROGRAM = True
MAX_SPECTROGRAM_SAVE  = 20   # 저장할 그림 개수 제한 (전체 저장하려면 None)

# ============================================================
#  ViSQOL / ODG (선택적 — 설치돼 있을 때만 계산, 없으면 자동 스킵)
# ============================================================
try:
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2

    _config = visqol_config_pb2.VisqolConfig()
    _config.audio.sample_rate = SAMPLE_RATE
    _config.options.use_speech_scoring = False
    _config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", "libsvm_nu_svr_model.txt"
    )
    _VISQOL_API = visqol_lib_py.VisqolApi()
    _VISQOL_API.Create(_config)
    VISQOL_AVAILABLE = True
except Exception as e:
    VISQOL_AVAILABLE = False
    print(f"⚠️  ViSQOL 미사용 (미설치 또는 이 환경에서 API 불일치): {e}")
    print("    Windows에는 공식 wheel이 없음 → WSL/Linux에서 `pip install visqol` 필요")


def compute_visqol(ref, deg, sr):
    if not VISQOL_AVAILABLE:
        return None
    try:
        result = _VISQOL_API.Run(ref.astype(np.float64), deg.astype(np.float64))
        return result.moslqo
    except Exception as e:
        print(f"    ViSQOL 계산 실패: {e}")
        return None


def compute_odg(ref, deg, sr):
    """
    PEAQ 기반 ODG. 유지보수되는 표준 파이썬 패키지가 없어 기본 비활성화.
    gstpeaq 등 외부 CLI 도구가 설치돼 있다면 subprocess 연동 코드를
    이 함수 안에 채워 넣으면 됨 (현재는 항상 None 반환).
    """
    return None


# ============================================================
#  지표 계산 함수
# ============================================================
def load_audio(path):
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.clip(audio.astype(np.float32), -1.0, 1.0), sr


def si_snr(reference, estimate, eps=1e-8):
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    s_target = (np.sum(estimate * reference) / (np.sum(reference ** 2) + eps)) * reference
    e_noise = estimate - s_target
    ratio = (np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps)
    return 10 * np.log10(ratio + eps)


def normalized_correlation(bits_a, bits_b):
    # dtype을 먼저 float64로 바꾼 뒤 연산해야 함 (uint8 상태로 *2-1 하면 0비트에서 언더플로우 발생)
    a = bits_a.astype(np.float64) * 2 - 1   # {0,1} -> {-1,+1}
    b = bits_b.astype(np.float64) * 2 - 1
    return float(np.sum(a * b) / len(a))


def decode_and_measure(detector, wm_audio, sr):
    x = torch.tensor(wm_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        detect_prob, message = detector.detect_watermark(
            x, sample_rate=sr,
            message_threshold=0.5,
            detection_threshold=0.5,
        )
    decoded_bits = message.squeeze(0).numpy().astype(np.int32)
    original_bits = FIXED_MESSAGE.squeeze(0).numpy().astype(np.int32)

    ber = float(np.mean(decoded_bits != original_bits))
    nc = normalized_correlation(original_bits, decoded_bits)
    dsr = 1 if detect_prob.item() >= DETECTION_THRESHOLD else 0
    return ber, nc, dsr, detect_prob.item()


def save_diff_spectrogram(orig, wm, sr, out_path):
    n_fft, hop = 1024, 256
    S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig, n_fft=n_fft, hop_length=hop)), ref=np.max)
    S_wm = librosa.amplitude_to_db(np.abs(librosa.stft(wm, n_fft=n_fft, hop_length=hop)), ref=np.max)
    S_diff = S_wm - S_orig

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, S, title, cmap in zip(
        axes, [S_orig, S_wm, S_diff],
        ["원본", "워터마크 삽입", "차이 (WM - 원본)"],
        ["magma", "magma", "coolwarm"],
    ):
        img = librosa.display.specshow(S, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=ax, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
#  메인
# ============================================================
def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_DIFF_SPECTROGRAM:
        SPEC_DIR.mkdir(parents=True, exist_ok=True)

    print("AudioSeal 탐지기 로딩 중...")
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    detector.eval()
    print("완료!\n")

    orig_files = sorted(ORIGINAL_DIR.rglob("*.wav"))
    print(f"원본 파일 수: {len(orig_files)}개")
    print("=" * 60)

    rows = []
    spec_saved = 0

    for i, orig_path in enumerate(orig_files):
        rel = orig_path.relative_to(ORIGINAL_DIR)
        wm_path = WM_DIR / rel.parent / f"wm_{rel.name}"

        if not wm_path.exists():
            print(f"[{i+1}] ⚠️  워터마크 파일 없음 (skip): {rel}")
            continue

        try:
            orig, sr = load_audio(orig_path)
            wm, _ = load_audio(wm_path)
            min_len = min(len(orig), len(wm))
            orig, wm = orig[:min_len], wm[:min_len]

            ber, nc, dsr, detect_prob = decode_and_measure(detector, wm, sr)
            sisnr = si_snr(orig, wm)
            visqol_score = compute_visqol(orig, wm, sr)
            odg_score = compute_odg(orig, wm, sr)

            spec_path = ""
            if SAVE_DIFF_SPECTROGRAM and (MAX_SPECTROGRAM_SAVE is None or spec_saved < MAX_SPECTROGRAM_SAVE):
                spec_out = SPEC_DIR / f"{rel.stem}_diff.png"
                save_diff_spectrogram(orig, wm, sr, spec_out)
                spec_path = str(spec_out)
                spec_saved += 1

            rows.append({
                "file": str(rel),
                "BER": round(ber, 4),
                "NC": round(nc, 4),
                "DSR": dsr,
                "detect_prob": round(detect_prob, 4),
                "SI-SNR(dB)": round(sisnr, 2),
                "ViSQOL": round(visqol_score, 3) if visqol_score is not None else "",
                "ODG": round(odg_score, 3) if odg_score is not None else "",
                "diff_spectrogram": spec_path,
            })

            print(f"[{i+1}/{len(orig_files)}] {rel} | "
                  f"BER={ber:.4f} NC={nc:.4f} DSR={dsr} SI-SNR={sisnr:.2f}dB")

        except Exception as e:
            print(f"[{i+1}] 💥 오류: {rel} → {e}")

    # CSV 저장
    fieldnames = ["file", "BER", "NC", "DSR", "detect_prob", "SI-SNR(dB)", "ViSQOL", "ODG", "diff_spectrogram"]
    csv_path = RESULT_DIR / "step2_noattack_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 요약
    print("\n" + "=" * 60)
    if rows:
        def avg(key):
            vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
            return np.mean(vals) if vals else float("nan")

        print(f"  파일 수     : {len(rows)}")
        print(f"  평균 BER    : {avg('BER'):.4f}")
        print(f"  평균 NC     : {avg('NC'):.4f}")
        print(f"  DSR         : {avg('DSR')*100:.1f}%")
        print(f"  평균 SI-SNR : {avg('SI-SNR(dB)'):.2f} dB")
        if VISQOL_AVAILABLE:
            print(f"  평균 ViSQOL : {avg('ViSQOL'):.3f}")
        else:
            print(f"  ViSQOL      : 미계산 (미설치)")
        print(f"  ODG         : 미계산 (외부 PEAQ 도구 필요)")
        print(f"  결과 CSV    : {csv_path}")
        if SAVE_DIFF_SPECTROGRAM:
            print(f"  차이 스펙트로그램 ({spec_saved}개): {SPEC_DIR}")
    else:
        print("  처리된 파일이 없습니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
