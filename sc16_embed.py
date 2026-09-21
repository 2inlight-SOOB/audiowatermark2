"""
embed_sc16.py
=============
SilentCipher SC-16 (16kHz, 32비트) 워터마크 삽입 스크립트
# 1. silentcipher 폴더로 이동

# 2. venv 생성
py -3.10 -m venv venv
# 3. venv 활성화
venv\Scripts\activate

# torch 먼저 설치 (2.0.0으로 고정)
pip install torch==2.0.0 silentcipher librosa soundfile numpy pystoi matplotlib

실행: python embed_sc16.py
폴더명 입력 예시: A01_30
"""

"""
embed_sc16.py
SilentCipher SC-16 watermark embedding script
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import silentcipher

BASE         = Path(r"C:\Users\user\OneDrive\Desktop\0708new_watermark")
RAW_WAV_BASE = BASE / "original"
WM_WAV_BASE  = BASE / "sc16_embed"
META_BASE    = BASE / "sc16_embed_meta"

SAMPLE_RATE  = 16000
DEVICE       = "cpu"
MODEL_TYPE   = "16k"

# 0~2 사이 정수 15개 (SC-16 모델 요구 형식)
WM_MESSAGE   = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
WM_MESSAGE_BYTES = [24, 97, 134]


def load_wav(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.clip(y.astype(np.float32), -1.0, 1.0)


def compute_ber(original: list, decoded: list) -> float:
    """복호된 8비트 바이트 메시지 기준 BER."""
    if decoded is None or len(decoded) != len(original):
        return 1.0
    original_bits = "".join(f"{value:08b}" for value in original)
    decoded_bits = "".join(f"{value:08b}" for value in decoded)
    errors = sum(a != b for a, b in zip(original_bits, decoded_bits))
    return errors / len(original_bits)


def encode_sc16(model, y: np.ndarray) -> tuple:
    """encode_wav 내부 binary_encode를 우회해서 직접 삽입."""
    with torch.no_grad():
        original_power = np.mean(y ** 2)
        if original_power == 0:
            return y, 0

        y_norm = y * np.sqrt(model.average_energy_VCTK / original_power)
        y_tensor = torch.FloatTensor(y_norm).unsqueeze(0).unsqueeze(0).to(model.device)

        carrier, carrier_phase = model.stft.transform(y_tensor.squeeze(1))
        carrier = carrier[:, None]
        carrier_phase = carrier_phase[:, None]

        msgs, _ = model.letters_encoding(carrier.shape[3], [WM_MESSAGE])
        msg_enc = torch.from_numpy(msgs[None]).to(model.device).float()

        carrier_enc = model.enc_c(carrier)
        msg_enc = model.enc_c.transform_message(msg_enc)

        merged_enc = torch.cat(
            (carrier_enc, carrier.repeat(1, 32, 1, 1), msg_enc.repeat(1, 32, 1, 1)), dim=1
        )

        message_info = model.dec_c(merged_enc, model.config.message_sdr)

        if model.config.utterance_level_normalization:
            message_info = message_info * (torch.mean((carrier ** 2), dim=(2, 3), keepdim=True) ** 0.5)

        if model.config.ensure_negative_message:
            message_info = -message_info
            carrier_reconst = torch.nn.functional.relu(message_info + carrier)
        else:
            carrier_reconst = torch.abs(message_info + carrier)

        model.stft.num_samples = y_tensor.shape[2]
        y_out = model.stft.inverse(carrier_reconst.squeeze(1), carrier_phase.squeeze(1))
        y_out = y_out.data.cpu().numpy()[0, 0]
        y_out = y_out * np.sqrt(original_power / model.average_energy_VCTK)

        sdr = float(model.sdr(y, y_out))
        return y_out.astype(np.float32), sdr


def decode_sc16(model, y: np.ndarray) -> dict:
    """복호 후 결과 반환."""
    result = model.decode_wav(y, SAMPLE_RATE, phase_shift_decoding=False)
    detected   = result["status"]
    decoded_msg = result["messages"][0]    if result["messages"]    else None
    confidence  = result["confidences"][0] if result["confidences"] else 0.0
    ber = compute_ber(WM_MESSAGE_BYTES, decoded_msg) if decoded_msg else 1.0
    return {"detected": detected, "decoded_msg": decoded_msg,
            "confidence": float(confidence), "ber": ber}


def main():
    print("=" * 60)
    print("  SilentCipher SC-16 워터마크 삽입 - 전체 자동 처리")
    print("=" * 60)

    print("[INFO] SC-16 모델 로딩 중...")
    model = silentcipher.get_model(model_type=MODEL_TYPE, device=DEVICE)
    print("[INFO] 모델 로딩 완료.\n")

    wav_files = sorted(RAW_WAV_BASE.rglob("*.wav"))
    if not wav_files:
        print(f"[ERROR] WAV 없음: {RAW_WAV_BASE}")
        return

    print(f"[INFO] 처리할 원본 WAV: {len(wav_files)}개")
    success_count, fail_count = 0, 0

    for fpath in wav_files:
        relative = fpath.relative_to(RAW_WAV_BASE)
        wm_path = WM_WAV_BASE / relative.parent / f"wm_{relative.name}"
        meta_path = META_BASE / relative.parent / relative.with_suffix(".json").name
        wm_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            y = load_wav(fpath)
            encoded, sdr = encode_sc16(model, y)

            dec = decode_sc16(model, encoded)
            sf.write(str(wm_path), encoded, SAMPLE_RATE, subtype="PCM_16")

            meta = {
                "filename":          str(relative),
                "model":             "SC-16",
                "sample_rate":       SAMPLE_RATE,
                "wm_message_symbols": WM_MESSAGE,
                "wm_message_bytes":   WM_MESSAGE_BYTES,
                "sdr_db":             round(sdr, 4),
                "verify_status":     dec["detected"],
                "verify_message":    dec["decoded_msg"],
                "verify_ber":        round(dec["ber"], 6),
                "verify_confidence": round(dec["confidence"], 6),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            flag = "O" if dec["detected"] else "X"
            print(f"[{success_count + fail_count + 1}/{len(wav_files)}] [{flag}] "
                  f"{relative} SDR={sdr:.2f}dB  BER={dec['ber']:.4f} "
                  f"conf={dec['confidence']:.4f}")

            if dec["detected"]:
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            import traceback
            print(f"[ERROR] {relative} - {e}")
            traceback.print_exc()
            fail_count += 1

    print(f"\n[완료] 성공 {success_count}개 / 실패 {fail_count}개")

    print("\n" + "="*60)
    print("[전체 완료]")


if __name__ == "__main__":
    main()