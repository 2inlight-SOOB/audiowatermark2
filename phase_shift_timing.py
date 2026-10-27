"""
phase_shift_timing.py
=====================
SilentCipher 디코딩에서 phase_shift_decoding=False vs True 의 '디코딩 시간'을 비교한다.
(리뷰의 A항목: phase_shift=True 전환 전 시간 측정)

왜 필요한가
-----------
- SC-16/SC-44 pipeline_full.py 는 공정 비교를 위해 PHASE_SHIFT_DECODING=True 로 바꿔 두었다.
- 공식 문서는 True 가 디코딩 시간을 크게 늘릴 수 있다고 경고하므로, 전체 재실행 전에
  실제로 몇 배 느려지는지 이 스크립트로 먼저 확인한다.
- 너무 느리면 pipeline_full.py 의 PHASE_SHIFT_DECODING 를 False 로 되돌리거나,
  desync(Crop/TSM) 공격에만 True 를 적용하는 식으로 조정하면 된다.

실행 (silentcipher 가 설치된 서버/환경에서)
--------------------------------------------
    python phase_shift_timing.py                 # 기본 경로, 각 모델 5개 파일
    python phase_shift_timing.py --n 10 --device cuda
    python phase_shift_timing.py --dir16 /path/sc16_embed --dir44 /path/sc44_embed

주의: 이 PC(Windows 로컬)에는 silentcipher 가 설치돼 있지 않다. 실제 실험이 도는
서버(torch==2.0.0 + silentcipher)에서 실행할 것.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import librosa
import silentcipher

# pipeline_full.py 와 동일한 기본 경로/설정
AUDIO_ROOT = Path("/home/hansung1/.vscode-server/audio")

CONFIGS = [
    {"tag": "SC-16", "model_type": "16k",   "sr": 16000, "dir": AUDIO_ROOT / "sc16_embed"},
    {"tag": "SC-44", "model_type": "44.1k", "sr": 44100, "dir": AUDIO_ROOT / "sc44_embed"},
]


def time_decode(model, y, sr, phase_shift, repeat):
    """repeat 회 디코딩의 평균 소요 시간(초)."""
    # 워밍업 1회 (lazy init / 캐시 제외)
    model.decode_wav(y, sr, phase_shift_decoding=phase_shift)
    t0 = time.perf_counter()
    for _ in range(repeat):
        model.decode_wav(y, sr, phase_shift_decoding=phase_shift)
    return (time.perf_counter() - t0) / repeat


def run_config(cfg, n, device, repeat, total_files, decodes_per_file):
    wavs = sorted(Path(cfg["dir"]).rglob("*.wav"))[:n]
    if not wavs:
        print(f"[{cfg['tag']}] WAV 없음: {cfg['dir']}  (건너뜀)")
        return

    print(f"\n[{cfg['tag']}] 모델 로딩 (model_type={cfg['model_type']}, device={device}) ...")
    model = silentcipher.get_model(model_type=cfg["model_type"], device=device)
    print(f"[{cfg['tag']}] 파일 {len(wavs)}개로 측정 (반복 {repeat}회/파일)")
    print(f"  {'file':<40} {'len(s)':>7} {'False(s)':>10} {'True(s)':>10} {'x배':>7}")

    f_times, t_times = [], []
    for w in wavs:
        y, _ = librosa.load(str(w), sr=cfg["sr"], mono=True)
        y = np.clip(y.astype(np.float32), -1.0, 1.0)
        tf = time_decode(model, y, cfg["sr"], False, repeat)
        tt = time_decode(model, y, cfg["sr"], True, repeat)
        f_times.append(tf)
        t_times.append(tt)
        ratio = tt / tf if tf > 0 else float("nan")
        print(f"  {w.name[:40]:<40} {len(y)/cfg['sr']:>7.1f} {tf:>10.3f} {tt:>10.3f} {ratio:>7.1f}")

    mf, mt = float(np.mean(f_times)), float(np.mean(t_times))
    print(f"  {'-'*76}")
    print(f"  {'평균':<40} {'':>7} {mf:>10.3f} {mt:>10.3f} {mt/mf if mf>0 else float('nan'):>7.1f}")

    # ---- 핵심: '배수'가 아니라 전체 실행 시간으로 판단한다 ----
    # 한 번의 전체 재실행에서 이 모델이 수행하는 디코딩 횟수 = 파일 수 x (공격 조건 수 + NoAttack)
    n_decodes = total_files * decodes_per_file
    proj_false_h = mf * n_decodes / 3600.0
    proj_true_h  = mt * n_decodes / 3600.0
    print(f"  => {cfg['tag']}: True가 평균 {mt/mf:.1f}배 느림(파일당 +{mt-mf:.2f}s).")
    print(f"     추정 전체 디코딩 시간(이 모델, {total_files}파일 x {decodes_per_file}디코딩="
          f"{n_decodes}회):")
    print(f"        False ≈ {proj_false_h:.1f}시간   /   True ≈ {proj_true_h:.1f}시간")
    print(f"     ※ 임베딩/공격/품질지표(ViSQOL 등) 시간은 별도이며 이 추정에 포함되지 않음.")
    return mt, n_decodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="모델별 측정 파일 수")
    ap.add_argument("--repeat", type=int, default=3, help="파일당 디코딩 반복 횟수(평균)")
    ap.add_argument("--device", default="cpu", help="cpu 또는 cuda")
    ap.add_argument("--total-files", type=int, default=150,
                    help="전체 재실행 시 모델당 처리할 파일 수 (데이터 확정 후 실제 값으로)")
    ap.add_argument("--decodes-per-file", type=int, default=15,
                    help="파일당 디코딩 횟수 = 공격 조건 수 + NoAttack (현재 pipeline_full 기준 약 15)")
    ap.add_argument("--dir16", default=None, help="SC-16 임베딩 WAV 폴더 (기본: AUDIO_ROOT/sc16_embed)")
    ap.add_argument("--dir44", default=None, help="SC-44 임베딩 WAV 폴더 (기본: AUDIO_ROOT/sc44_embed)")
    args = ap.parse_args()

    if args.dir16:
        CONFIGS[0]["dir"] = Path(args.dir16)
    if args.dir44:
        CONFIGS[1]["dir"] = Path(args.dir44)

    print("=" * 80)
    print("  SilentCipher phase_shift_decoding  False vs True  디코딩 시간 비교")
    print("=" * 80)
    total_true_h = 0.0
    for cfg in CONFIGS:
        res = run_config(cfg, args.n, args.device, args.repeat,
                         args.total_files, args.decodes_per_file)
        if res:
            mt, n_dec = res
            total_true_h += mt * n_dec / 3600.0

    print("\n" + "=" * 80)
    print(f"  두 모델 합계 True 디코딩 추정 ≈ {total_true_h:.1f}시간 (SC-16 + SC-44)")
    print("=" * 80)
    print("판단 기준: '배수'가 아니라 '전체 실행 시간이 서버에서 감당되는가'로 정한다.")
    print(" - 5배가 나와도 전체가 (예: 하루 안에) 끝나면 True 유지가 맞다. 공정성 때문.")
    print(" - False로 되돌리면 공정성 문제가 그대로 돌아와 논문에서 따로 설명해야 한다.")
    print(" - 위 합계가 서버에서 감당 불가일 때만 PHASE_SHIFT_DECODING=False 로 두고,")
    print("   시간축(Crop/TSM)·코덱(Opus/AAC/MP3) 공격에만 True 를 적용하는 분기로 바꾼 뒤")
    print("   그 사실을 논문에 명시한다.")


if __name__ == "__main__":
    main()
