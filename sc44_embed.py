"""
sc44_embed.py
=============
SilentCipher SC-44 (44.1kHz, 40비트) 워터마크 삽입 스크립트

입력:  original/ 아래의 모든 WAV 파일
출력:  sc44_embed/<원본 상대 폴더>/wm_<원본 파일명>.wav
메타:  sc44_embed_meta/<원본 상대 폴더>/<원본 파일명>.json

주의: 입력 오디오는 SC-44 기준인 44.1kHz mono로 자동 변환됩니다.
실행: python sc44_embed.py
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import silentcipher

# ─────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────

BASE         = Path(r"C:\Users\user\OneDrive\Desktop\0708new_watermark")
RAW_WAV_BASE = BASE / "original"
WM_WAV_BASE  = BASE / "sc44_embed"
META_BASE    = BASE / "sc44_embed_meta"

SAMPLE_RATE  = 44100          # SC-44 기준
DEVICE       = "cpu"
MODEL_TYPE   = "44.1k"

# SC-44: 5개 × 8비트 = 40비트
WM_MESSAGE   = [123, 234, 111, 222, 11]

# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def load_wav(path: Path) -> np.ndarray:
    """원본 WAV를 44.1kHz mono로 리샘플링하여 로드."""
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.clip(y.astype(np.float32), -1.0, 1.0)


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


def main():
    print("=" * 60)
    print("  SilentCipher SC-44 워터마크 삽입 - 전체 자동 처리")
    print("  ※ original/ 전체 탐색 / 44.1kHz mono 변환 / PCM-32 저장")
    print("=" * 60)

    if not RAW_WAV_BASE.exists():
        print(f"[ERROR] 입력 폴더를 찾을 수 없습니다: {RAW_WAV_BASE}")
        return

    wav_files = sorted(RAW_WAV_BASE.rglob("*.wav"))
    if not wav_files:
        print(f"[ERROR] WAV 파일이 없습니다: {RAW_WAV_BASE}")
        return

    print(f"\n처리할 원본 WAV: {len(wav_files)}개")
    print(f"입력          : {RAW_WAV_BASE}")
    print(f"출력          : {WM_WAV_BASE}")
    print(f"메타데이터    : {META_BASE}")
    print(f"메시지        : {WM_MESSAGE} (40비트)\n")

    print("[INFO] SC-44 모델 로딩 중...")
    model = silentcipher.get_model(model_type=MODEL_TYPE, device=DEVICE)
    print("[INFO] 모델 로딩 완료.\n")

    success_count, fail_count = 0, 0

    for fpath in wav_files:
        relative = fpath.relative_to(RAW_WAV_BASE)
        wm_path = WM_WAV_BASE / relative.parent / f"wm_{relative.name}"
        meta_path = META_BASE / relative.parent / relative.with_suffix(".json").name
        wm_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            y = load_wav(fpath)
            encoded, sdr = model.encode_wav(y, SAMPLE_RATE, WM_MESSAGE)

            result      = model.decode_wav(encoded, SAMPLE_RATE, phase_shift_decoding=False)
            detected    = result["status"]
            decoded_msg = result["messages"][0]    if result["messages"]    else None
            confidence  = result["confidences"][0] if result["confidences"] else 0.0
            ber = compute_ber(WM_MESSAGE, decoded_msg) if decoded_msg else 1.0

            sf.write(str(wm_path), encoded, SAMPLE_RATE, subtype="PCM_32")

            meta = {
                "filename":          str(relative),
                "model":             "SC-44",
                "sample_rate":       SAMPLE_RATE,
                "message_bits":      40,
                "wm_message_bytes":  WM_MESSAGE,
                "sdr_db":            round(float(sdr), 4) if sdr is not None else None,
                "verify_status":     detected,
                "verify_message":    decoded_msg,
                "verify_ber":        round(ber, 6),
                "verify_confidence": round(float(confidence), 6),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            flag = "O" if detected else "X"
            print(f"[{success_count + fail_count + 1}/{len(wav_files)}] [{flag}] "
                  f"{relative} SDR={sdr:.2f}dB  BER={ber:.4f}  conf={confidence:.4f}")
            if detected:
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"[ERROR] {relative} - {e}")
            fail_count += 1

    print()
    print(f"[DONE] 성공 {success_count}개 / 실패 {fail_count}개 / 전체 {len(wav_files)}개")
    print(f"       워터마크 WAV → {WM_WAV_BASE}")
    print(f"       메타데이터  → {META_BASE}")


if __name__ == "__main__":
    main()
