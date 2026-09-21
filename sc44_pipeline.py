"""
sc44_pipeline.py
================
SilentCipher SC-44 (44.1kHz, 40비트) 통합 실험 파이프라인

실행: python sc44_pipeline.py
대상: sc44_embed/ 아래의 모든 워터마크 WAV

주의: SC-44는 44.1kHz 기준으로 동작합니다.
      WavMark / AudioSeal / SC-16과 샘플링 레이트가 다르므로
      실험 결과 비교 시 논문에 명시 필요.
"""

import os
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import silentcipher
from pystoi import stoi as calc_stoi
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────

BASE        = Path(r"C:\Users\user\OneDrive\Desktop\0708new_watermark")
WM_WAV_BASE = BASE / "sc44_embed"
RESULT_BASE = BASE / "sc44_pipeline_results"

SAMPLE_RATE  = 44100
DEVICE       = "cpu"
MODEL_TYPE   = "44.1k"
WM_MESSAGE   = [123, 234, 111, 222, 11]   # 5개 × 8비트 = 40비트
MESSAGE_BITS = 40

MP3_BITRATES     = [128, 96, 64, 48, 32]
AWGN_SNRS        = [50, 40, 30, 20, 10]
COMBO_CONDITIONS = [
    {"mp3_kbps": 64, "snr_db": 30},
    {"mp3_kbps": 32, "snr_db": 20},
]
DSR_THRESHOLD = 95.0

np.random.seed(42)

# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def load_wav(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.clip(y.astype(np.float32), -1.0, 1.0)


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    sig_power   = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise       = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return np.clip((signal + noise).astype(np.float32), -1.0, 1.0)


def apply_mp3(wav: np.ndarray, kbps: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        sf.write(str(t / "in.wav"), wav, SAMPLE_RATE, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-i", str(t / "in.wav"),
                        "-b:a", f"{kbps}k", str(t / "out.mp3")],
                       capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(t / "out.mp3"), str(t / "out.wav")],
                       capture_output=True, check=True)
        out, _ = librosa.load(str(t / "out.wav"), sr=SAMPLE_RATE, mono=True)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def compute_ber(original: list, decoded: list) -> float:
    if decoded is None or len(decoded) != len(original):
        return 1.0

    orig_bits, deco_bits = [], []
    for o, d in zip(original, decoded):
        for bit in range(8):
            orig_bits.append((o >> bit) & 1)
            deco_bits.append((d >> bit) & 1)
    errors = sum(ob != db for ob, db in zip(orig_bits, deco_bits))
    return errors / len(orig_bits)


def decode_one(model, attacked: np.ndarray) -> dict:
    result   = model.decode_wav(attacked, SAMPLE_RATE, phase_shift_decoding=False)
    detected = result["status"]
    msg      = result["messages"][0]    if result["messages"]    else None
    conf     = result["confidences"][0] if result["confidences"] else 0.0
    ber      = compute_ber(WM_MESSAGE, msg) if msg else 1.0
    return {"detected": detected, "ber": round(ber, 6), "confidence": round(float(conf), 6)}


def get_stoi(original: np.ndarray, degraded: np.ndarray) -> float:
    try:
        return round(float(calc_stoi(original, degraded, SAMPLE_RATE, extended=False)), 6)
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────
# 공격 실행
# ─────────────────────────────────────────────

def run_attack(model, wav_files: list, attack_fn, label: str) -> dict:
    per_file = []
    for fpath in wav_files:
        original = load_wav(fpath)
        attacked = attack_fn(original)
        scores   = decode_one(model, attacked)
        stoi_v   = get_stoi(original, attacked)

        per_file.append({
            "fname": fpath.name, "ber": scores["ber"],
            "detected": scores["detected"], "stoi": stoi_v,
            "confidence": scores["confidence"],
        })

        flag = "O" if scores["detected"] else "X"
        print(f"    [{flag}] {fpath.name:<35s} "
              f"BER={scores['ber']:.4f}  STOI={stoi_v:.4f}  conf={scores['confidence']:.4f}")

    n         = len(per_file)
    dsr       = sum(r["detected"] for r in per_file) / n * 100
    mean_ber  = float(np.mean([r["ber"]  for r in per_file]))
    mean_stoi = float(np.nanmean([r["stoi"] for r in per_file]))
    status    = "OK" if dsr >= DSR_THRESHOLD else "FAIL"
    icon      = "[OK]" if status == "OK" else "[FAIL]"

    print(f"  {icon} [{label}] 요약 → "
          f"DSR={dsr:.2f}%  BER={mean_ber:.4f}  STOI={mean_stoi:.4f}  ({status})")

    return {"label": label, "per_file": per_file,
            "dsr": round(dsr, 2), "mean_ber": round(mean_ber, 6),
            "mean_stoi": round(mean_stoi, 6), "status": status}


# ─────────────────────────────────────────────
# 저장 + 시각화
# ─────────────────────────────────────────────

def save_results(all_rows: list, folder_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{folder_name}_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "dsr", "mean_ber", "mean_stoi", "status"])
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r[k] for k in ["label", "dsr", "mean_ber", "mean_stoi", "status"]})
    print(f"[SAVED] 요약 CSV  → {csv_path}")

    json_path = out_dir / f"{folder_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] 요약 JSON → {json_path}")

    detail_path = out_dir / f"{folder_name}_detail.csv"
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "fname", "ber", "detected", "stoi", "confidence"])
        writer.writeheader()
        for r in all_rows:
            for pf in r["per_file"]:
                writer.writerow({"condition": r["label"], **pf})
    print(f"[SAVED] 상세 CSV  → {detail_path}")


def plot_results(all_rows: list, folder_name: str, out_dir: Path):
    labels = [r["label"] for r in all_rows]
    dsrs   = [r["dsr"]       for r in all_rows]
    bers   = [r["mean_ber"]  for r in all_rows]
    stois  = [r["mean_stoi"] for r in all_rows]
    x      = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(labels) * 0.75), 12))
    fig.suptitle(f"SilentCipher SC-44 (40bit) Robustness — {folder_name}",
                 fontsize=13, fontweight="bold")

    colors = ["#2ecc71" if d >= DSR_THRESHOLD else "#e74c3c" for d in dsrs]
    bars = axes[0].bar(x, dsrs, color=colors)
    axes[0].axhline(DSR_THRESHOLD, color="gray", linestyle="--", linewidth=1,
                    label=f"Threshold ({DSR_THRESHOLD}%)")
    axes[0].set_ylabel("DSR (%)"); axes[0].set_ylim(0, 110)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].legend(fontsize=8)
    for bar, v in zip(bars, dsrs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    axes[1].bar(x, bers, color="#3498db")
    axes[1].set_ylabel("Mean BER"); axes[1].set_ylim(0, max(bers) * 1.25 + 0.01)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    for i, v in enumerate(bers):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    axes[2].bar(x, stois, color="#9b59b6")
    axes[2].set_ylabel("Mean STOI"); axes[2].set_ylim(0, 1.08)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    for i, v in enumerate(stois):
        axes[2].text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    png_path = out_dir / f"{folder_name}_sc44_result.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] 그래프   → {png_path}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SilentCipher SC-44 (40비트) 통합 실험 파이프라인")
    print("  sc44_embed/ 전체 WAV를 대상으로 자동 처리")
    print("  ※ 44.1kHz 기준 / 입력 파일은 이미 SC-44 형식으로 저장됨")
    print("=" * 60)

    if not WM_WAV_BASE.exists():
        print(f"[ERROR] 임베딩 폴더를 찾을 수 없습니다: {WM_WAV_BASE}")
        print("        먼저 sc44_embed.py를 실행하세요.")
        return

    wav_files = sorted(WM_WAV_BASE.rglob("*.wav"))
    if not wav_files:
        print(f"[ERROR] WAV 파일이 없습니다: {WM_WAV_BASE}")
        return

    print(f"\n대상 폴더 : {WM_WAV_BASE}")
    print(f"총 파일 수: {len(wav_files)}개")
    print(f"모델      : SC-44 / {MESSAGE_BITS}비트 / {SAMPLE_RATE}Hz\n")

    print("[INFO] SC-44 모델 로딩 중...")
    model = silentcipher.get_model(model_type=MODEL_TYPE, device=DEVICE)
    print("[INFO] 모델 로딩 완료.\n")

    folder_name = "all"
    out_dir  = RESULT_BASE
    all_rows = []

    print("[STEP 1] MP3 압축 공격")
    print("=" * 60)
    for kbps in MP3_BITRATES:
        label = f"MP3_{kbps}kbps"
        print(f"\n▶ {label}")
        all_rows.append(run_attack(model, wav_files,
                                   attack_fn=lambda w, k=kbps: apply_mp3(w, k),
                                   label=label))

    print("\n[STEP 2] AWGN 공격")
    print("=" * 60)
    for snr in AWGN_SNRS:
        label = f"AWGN_SNR{snr}dB"
        print(f"\n▶ {label}")
        all_rows.append(run_attack(model, wav_files,
                                   attack_fn=lambda w, s=snr: add_awgn(w, s),
                                   label=label))

    print("\n[STEP 3] 복합 공격 (MP3 → AWGN)")
    print("=" * 60)
    for cond in COMBO_CONDITIONS:
        kbps  = cond["mp3_kbps"]; snr = cond["snr_db"]
        label = f"MP3_{kbps}k+SNR{snr}dB"
        print(f"\n▶ {label}")
        all_rows.append(run_attack(model, wav_files,
                                   attack_fn=lambda w, k=kbps, s=snr: add_awgn(apply_mp3(w, k), s),
                                   label=label))

    print("\n[STEP 4] 결과 저장 및 시각화")
    print("=" * 60)
    save_results(all_rows, folder_name, out_dir)
    plot_results(all_rows, folder_name, out_dir)

    print(f"\n{'=' * 65}")
    print(f"[최종 요약] SC-44 (40비트) - {folder_name}")
    print(f"{'조건':<28} {'DSR(%)':>8} {'BER':>8} {'STOI':>7} {'판정':>6}")
    print("-" * 65)
    for r in all_rows:
        flag = "OK  " if r["status"] == "OK" else "FAIL"
        print(f"{r['label']:<28} {r['dsr']:>8.2f} {r['mean_ber']:>8.4f} {r['mean_stoi']:>7.4f}  {flag}")
    print("=" * 65)


if __name__ == "__main__":
    main()
