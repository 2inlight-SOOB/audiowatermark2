"""
wavmark_new_embed.py
=====================
WavMark (16kHz, 32비트 = 16비트 sync 패턴 + 16비트 payload) 워터마크 삽입 스크립트

입력:  original/ 아래의 모든 WAV 파일
출력:  wavmark_embed/<원본 상대 폴더>/wm_<원본 파일명>.wav
메타:  wavmark_embed_meta/<원본 상대 폴더>/<원본 파일명>.json

주의: 입력 오디오는 WavMark 기준인 16kHz mono로 자동 변환됩니다.
실행: python wavmark_new_embed.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import wavmark
from wavmark.utils import wm_add_util

# ─────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────

BASE         = Path(r"C:\Users\user\OneDrive\Desktop\0708new_watermark")
RAW_WAV_BASE = BASE / "original"
WM_WAV_BASE  = BASE / "wavmark_embed"
META_BASE    = BASE / "wavmark_embed_meta"

SAMPLE_RATE      = 16000          # WavMark 기준
PATTERN_BIT_LEN  = 16
WM_ID            = "2inlightSOOB-0001"

# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def load_wav(path: Path) -> np.ndarray:
    """원본 WAV를 16kHz mono로 리샘플링하여 로드."""
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.clip(y.astype(np.float32), -1.0, 1.0)


def make_payload(wm_id: str, n_bits: int = 16) -> np.ndarray:
    h = hashlib.sha256(wm_id.encode("utf-8")).digest()
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8)).astype(np.uint8)
    return bits[:n_bits]


def to_jsonable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


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


def main():
    print("=" * 60)
    print("  WavMark 워터마크 삽입 - 전체 자동 처리")
    print("  ※ original/ 전체 탐색 / 16kHz mono 변환 / PCM-16 저장")
    print("=" * 60)

    if not RAW_WAV_BASE.exists():
        print(f"[ERROR] 입력 폴더를 찾을 수 없습니다: {RAW_WAV_BASE}")
        return

    wav_files = sorted(RAW_WAV_BASE.rglob("*.wav"))
    if not wav_files:
        print(f"[ERROR] WAV 파일이 없습니다: {RAW_WAV_BASE}")
        return

    payload = make_payload(WM_ID, n_bits=PATTERN_BIT_LEN)
    pattern = list(wm_add_util.fix_pattern[0:PATTERN_BIT_LEN])
    original_32bit = pattern + payload.tolist()

    print(f"\n처리할 원본 WAV: {len(wav_files)}개")
    print(f"입력          : {RAW_WAV_BASE}")
    print(f"출력          : {WM_WAV_BASE}")
    print(f"메타데이터    : {META_BASE}")
    print(f"WM_ID         : {WM_ID}")
    print(f"payload       : {payload.tolist()} (16비트, 총 32비트)\n")

    print("[INFO] WavMark 모델 로딩 중...")
    model = wavmark.load_model()
    model.eval()
    print("[INFO] 모델 로딩 완료.\n")

    success_count, fail_count, skip_count = 0, 0, 0

    for fpath in wav_files:
        relative = fpath.relative_to(RAW_WAV_BASE)
        wm_path = WM_WAV_BASE / relative.parent / f"wm_{relative.name}"
        meta_path = META_BASE / relative.parent / relative.with_suffix(".json").name
        wm_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        progress = success_count + fail_count + skip_count + 1

        if wm_path.exists() and meta_path.exists():
            print(f"[{progress}/{len(wav_files)}] [SKIP] 이미 처리됨: {relative}", flush=True)
            skip_count += 1
            continue

        try:
            y = load_wav(fpath)
            if len(y) < SAMPLE_RATE // 2:
                print(f"[{progress}/{len(wav_files)}] [SKIP] 너무 짧음: {relative}", flush=True)
                fail_count += 1
                continue

            encoded, info = wavmark.encode_watermark(
                model, y, payload,
                pattern_bit_length=PATTERN_BIT_LEN,
                min_snr=20, max_snr=38,
                show_progress=False,
            )
            encoded = np.asarray(encoded).astype(np.float32).squeeze()
            encoded = np.clip(encoded, -1.0, 1.0)

            dec_payload, dec_info = wavmark.decode_watermark(
                model, encoded, len_start_bit=PATTERN_BIT_LEN, show_progress=False,
            )
            results     = dec_info.get("results", []) if isinstance(dec_info, dict) else []
            detected    = len(results) > 0
            decoded_msg = results[0]["msg"] if results else None
            confidence  = float(results[0]["sim"]) if results else 0.0
            ber = compute_ber(original_32bit, decoded_msg)

            sf.write(str(wm_path), encoded, SAMPLE_RATE, subtype="PCM_16")

            meta = {
                "filename":           str(relative),
                "model":              "WavMark",
                "sample_rate":        SAMPLE_RATE,
                "message_bits":       32,
                "wm_id":              WM_ID,
                "pattern_bit_length": PATTERN_BIT_LEN,
                "wm_message_bits":    original_32bit,
                "snr_db":             round(float(info.get("snr", float("nan"))), 4)
                                       if info.get("snr") is not None else None,
                "verify_status":      detected,
                "verify_message":     to_jsonable(decoded_msg),
                "verify_ber":         round(ber, 6),
                "verify_confidence":  round(confidence, 6),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            flag = "O" if detected else "X"
            print(f"[{progress}/{len(wav_files)}] [{flag}] "
                  f"{relative} BER={ber:.4f}  conf={confidence:.4f}", flush=True)
            if detected:
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"[ERROR] {relative} - {e}", flush=True)
            fail_count += 1

    print()
    print(f"[DONE] 성공 {success_count}개 / 실패 {fail_count}개 / "
          f"건너뜀(기처리) {skip_count}개 / 전체 {len(wav_files)}개")
    print(f"       워터마크 WAV → {WM_WAV_BASE}")
    print(f"       메타데이터  → {META_BASE}")


if __name__ == "__main__":
    main()
