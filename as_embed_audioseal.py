import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

from audioseal import AudioSeal

# ============================================================
#  설정
# ============================================================
INPUT_DIR  = Path(r"C:\Users\user\OneDrive\Desktop\new_watermark\original")
OUTPUT_DIR = Path(r"C:\Users\user\OneDrive\Desktop\new_watermark\embed")
SAMPLE_RATE = 16000

# 고정 메시지 16비트 (재현성 확보)
np.random.seed(42)
FIXED_MESSAGE = torch.tensor(
    [[1,0,1,1,0,0,1,0,1,1,0,1,0,1,0,0]], dtype=torch.float32
)  # shape: (1, 16)

# ============================================================
#  메인
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    print("AudioSeal 모델 로딩 중...")
    generator = AudioSeal.load_generator("audioseal_wm_16bits")
    generator.eval()
    print("완료!\n")

    # 전체 wav 파일 수집
    wav_files = sorted(INPUT_DIR.rglob("*.wav"))
    print(f"총 파일 수: {len(wav_files)}개")
    print("=" * 60)

    ok = 0
    skip = 0
    error = 0

    for i, p in enumerate(wav_files):
        try:
            # 오디오 로드
            audio, sr = sf.read(str(p), always_2d=False)

            # 모노 변환
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            audio = np.clip(audio, -1.0, 1.0)

            # sr 확인
            if sr != SAMPLE_RATE:
                print(f"[{i+1}] [WARN]  sr={sr} (skip): {p.name}")
                skip += 1
                continue

            # 길이 확인 (3~7초)
            duration = len(audio) / sr
            if duration < 3.0 or duration > 7.0:
                print(f"[{i+1}] [WARN]  길이 {duration:.1f}초 (skip): {p.name}")
                skip += 1
                continue

            # tensor 변환 (1, 1, T)
            audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0)

            # 워터마크 삽입
            with torch.no_grad():
                watermarked = generator(
                    audio_tensor,
                    message=FIXED_MESSAGE,
                    sample_rate=SAMPLE_RATE
                )

            # numpy 변환
            wm_audio = watermarked.squeeze().numpy()
            wm_audio = np.clip(wm_audio, -1.0, 1.0)

            # 저장 (폴더 구조 유지, 파일명에 wm_ 접두사 추가: A01_40/wm_LA_D_xxxx.wav)
            rel_path = p.relative_to(INPUT_DIR)
            out_path = OUTPUT_DIR / rel_path.parent / f"wm_{rel_path.name}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), wm_audio, SAMPLE_RATE, subtype="PCM_16")

            ok += 1
            print(f"[{i+1}/{len(wav_files)}] [OK] {rel_path}")

        except Exception as e:
            print(f"[{i+1}] [FAIL] 오류: {p.name} → {e}")
            error += 1

    print("\n" + "=" * 60)
    print(f"  완료: 성공={ok}, 스킵={skip}, 오류={error}")
    print(f"  저장 위치: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()