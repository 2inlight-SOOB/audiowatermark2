"""
wavmark_new_pipeline.py
========================
WavMark (16kHz, 32비트) 통합 실험 파이프라인

실행: python wavmark_new_pipeline.py
대상: wavmark_embed/ 아래의 모든 워터마크 WAV

as_pipeline.py(AudioSeal), sc44_pipeline.py/sc16_pipeline.py(SilentCipher)와 동일한 구조.
"""

import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from pystoi import stoi as calc_stoi
import matplotlib
import matplotlib.pyplot as plt

import wavmark
from wavmark.utils import wm_add_util

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────

BASE        = Path(r"C:\Users\user\OneDrive\Desktop\0708new_watermark")
WM_WAV_BASE = BASE / "wavmark_embed"

SAMPLE_RATE      = 16000
PATTERN_BIT_LEN  = 16
WM_ID            = "2inlightSOOB-0001"

# 테스트용: 각 파일을 앞부분 N초만 잘라서 처리 (WavMark 디코딩이 느려서 전체 길이는
# 오래 걸림). None이면 전체 길이 그대로 처리. 이미 N초보다 짧은 파일은 그대로 둠.
TEST_CLIP_SECONDS = 9
RESULT_BASE = BASE / ("wavmark_pipeline_results" if TEST_CLIP_SECONDS is None
                       else f"wavmark_pipeline_results_clip{TEST_CLIP_SECONDS}s")
STATE_PATH = BASE / "wavmark_pipeline_resume_state.json"

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
    if TEST_CLIP_SECONDS is not None:
        y = y[:int(SAMPLE_RATE * TEST_CLIP_SECONDS)]
    return np.clip(y.astype(np.float32), -1.0, 1.0)


def make_payload(wm_id: str, n_bits: int = 16) -> np.ndarray:
    import hashlib
    h = hashlib.sha256(wm_id.encode("utf-8")).digest()
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8)).astype(np.uint8)
    return bits[:n_bits]


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


def compute_ber(original: list, decoded) -> float:
    """payload 16비트(sync 패턴 제외) 기준 BER."""
    if decoded is None:
        return 1.0
    orig_payload = original[PATTERN_BIT_LEN:]
    deco_payload = list(decoded)[PATTERN_BIT_LEN:]
    if len(orig_payload) != len(deco_payload):
        return 1.0
    errors = sum(o != d for o, d in zip(orig_payload, deco_payload))
    return errors / len(orig_payload)


def decode_one(model, attacked: np.ndarray, original_32bit: list) -> dict:
    dec_payload, dec_info = wavmark.decode_watermark(
        model, attacked, len_start_bit=PATTERN_BIT_LEN, show_progress=False,
    )
    results     = dec_info.get("results", []) if isinstance(dec_info, dict) else []
    detected    = len(results) > 0
    decoded_msg = results[0]["msg"] if results else None
    ber         = compute_ber(original_32bit, decoded_msg)
    return {"detected": detected, "ber": round(ber, 6)}


def get_stoi(original: np.ndarray, degraded: np.ndarray) -> float:
    try:
        min_len = min(len(original), len(degraded))
        return round(float(calc_stoi(original[:min_len], degraded[:min_len],
                                     SAMPLE_RATE, extended=False)), 6)
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────
# 공격 실행
# ─────────────────────────────────────────────

def run_attack(model, wav_files: list, attack_fn, label: str, original_32bit: list) -> dict:
    per_file = []
    for fpath in wav_files:
        original = load_wav(fpath)
        attacked = attack_fn(original)
        scores   = decode_one(model, attacked, original_32bit)
        stoi_v   = get_stoi(original, attacked)

        per_file.append({
            "fname": fpath.name, "ber": scores["ber"],
            "detected": scores["detected"], "stoi": stoi_v,
        })

        flag = "O" if scores["detected"] else "X"
        print(f"    [{flag}] {fpath.name:<35s} "
              f"BER={scores['ber']:.4f}  STOI={stoi_v:.4f}")

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
        writer = csv.DictWriter(f, fieldnames=["condition", "fname", "ber", "detected", "stoi"])
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
    fig.suptitle(f"WavMark (32bit) Robustness — {folder_name}",
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
    png_path = out_dir / f"{folder_name}_wavmark_result.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] 그래프   → {png_path}")


# ─────────────────────────────────────────────
# 복원(재개) 기능
# ─────────────────────────────────────────────

def load_resume_state() -> tuple[list, list]:
    if not STATE_PATH.exists():
        return [], []
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_rows = data.get("all_rows", [])
        completed = data.get("completed_labels", [])
        return all_rows, completed
    except Exception:
        return [], []


def save_resume_state(all_rows: list, completed_labels: list):
    payload = {
        "all_rows": all_rows,
        "completed_labels": completed_labels,
    }
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def clear_resume_state():
    if STATE_PATH.exists():
        STATE_PATH.unlink()


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  WavMark (32비트) 통합 실험 파이프라인")
    print("  wavmark_embed/ 전체 WAV를 대상으로 자동 처리")
    print("  ※ 16kHz 기준 / 입력 파일은 이미 WavMark 형식으로 저장됨")
    print("  ※ 중간 종료 후 재시작 시 이어서 진행됩니다.")
    print("=" * 60)

    if not WM_WAV_BASE.exists():
        print(f"[ERROR] 임베딩 폴더를 찾을 수 없습니다: {WM_WAV_BASE}")
        print("        먼저 wavmark_new_embed.py를 실행하세요.")
        return

    wav_files = sorted(WM_WAV_BASE.rglob("*.wav"))
    if not wav_files:
        print(f"[ERROR] WAV 파일이 없습니다: {WM_WAV_BASE}")
        return

    payload = make_payload(WM_ID, n_bits=PATTERN_BIT_LEN)
    pattern = list(wm_add_util.fix_pattern[0:PATTERN_BIT_LEN])
    original_32bit = pattern + payload.tolist()

    print(f"\n대상 폴더 : {WM_WAV_BASE}")
    print(f"총 파일 수: {len(wav_files)}개")
    print(f"모델      : WavMark / 32비트(16 sync + 16 payload) / {SAMPLE_RATE}Hz\n")

    print("[INFO] WavMark 모델 로딩 중...")
    model = wavmark.load_model()
    model.eval()
    print("[INFO] 모델 로딩 완료.\n")

    folder_name = "all"
    out_dir  = RESULT_BASE
    all_rows, completed_labels = load_resume_state()
    completed_set = set(completed_labels)

    attack_specs = []
    for kbps in MP3_BITRATES:
        attack_specs.append({"type": "mp3", "label": f"MP3_{kbps}kbps", "fn": lambda w, k=kbps: apply_mp3(w, k)})
    for snr in AWGN_SNRS:
        attack_specs.append({"type": "awgn", "label": f"AWGN_SNR{snr}dB", "fn": lambda w, s=snr: add_awgn(w, s)})
    for cond in COMBO_CONDITIONS:
        kbps  = cond["mp3_kbps"]; snr = cond["snr_db"]
        attack_specs.append({"type": "combo", "label": f"MP3_{kbps}k+SNR{snr}dB",
                            "fn": lambda w, k=kbps, s=snr: add_awgn(apply_mp3(w, k), s)})

    print("[INFO] 재개 상태 확인")
    if completed_labels:
        print(f"  이미 완료된 조건: {', '.join(completed_labels)}")
    else:
        print("  아직 완료된 조건 없음. 처음부터 실행합니다.")

    for spec in attack_specs:
        label = spec["label"]
        if label in completed_set:
            print(f"[SKIP] 이미 완료: {label}")
            continue

        print(f"\n[STEP] ▶ {label}")
        row = run_attack(model, wav_files,
                         attack_fn=spec["fn"],
                         label=label, original_32bit=original_32bit)
        all_rows.append(row)
        completed_set.add(label)
        save_resume_state(all_rows, sorted(completed_set))
        print(f"[CHECKPOINT] 저장됨: {label} (재개용 상태 파일: {STATE_PATH.name})")

    print("\n[STEP 4] 결과 저장 및 시각화")
    print("=" * 60)
    save_results(all_rows, folder_name, out_dir)
    plot_results(all_rows, folder_name, out_dir)
    clear_resume_state()

    print(f"\n{'=' * 65}")
    print(f"[최종 요약] WavMark (32비트) - {folder_name}")
    print(f"{'조건':<28} {'DSR(%)':>8} {'BER':>8} {'STOI':>7} {'판정':>6}")
    print("-" * 65)
    for r in all_rows:
        flag = "OK  " if r["status"] == "OK" else "FAIL"
        print(f"{r['label']:<28} {r['dsr']:>8.2f} {r['mean_ber']:>8.4f} {r['mean_stoi']:>7.4f}  {flag}")
    print("=" * 65)


if __name__ == "__main__":
    main()
