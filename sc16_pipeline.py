r"""
실행 전 환경 설정
------------------
PowerShell에서 먼저 실행:

    cd "C:\Users\user\OneDrive\Desktop\0708new_watermark"
    .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install torch==2.0.0 silentcipher librosa soundfile numpy pystoi matplotlib
    python -m pip install visqol encodec descript-audio-codec
    python -c "import torch, silentcipher; print('SC-16 environment OK')"

실행:

    python sc16_pipeline.py

VS Code에서는 `Ctrl + Shift + P` → `Python: Select Interpreter`에서
위 패키지를 설치한 SC-16 가상환경(`venv`)을 선택하세요.

MP3/Opus 공격을 사용하려면 시스템에 `ffmpeg`도 설치되어 있어야 합니다.

`visqol`, `encodec`, `descript-audio-codec`는 전체 평가에 필요한 패키지입니다.
각각 ViSQOL 음질 점수, EnCodec 공격, DAC 공격에 사용됩니다.

SC-16 evaluation pipeline for the existing ``embed`` folder.

The input set is discovered from ``embed`` only. Matching files in ``original``
are used only as references for quality metrics and difference spectrograms.
"""

import csv
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import silentcipher
from pystoi import stoi as calc_stoi

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"C:\Users\user\OneDrive\Desktop\0708new_watermark")
ORIGINAL_DIR = ROOT / "original"
EMBED_DIR = ROOT / "sc16_embed"
RESULT_DIR = ROOT / "sc16_pipeline_results"
SPEC_DIR = RESULT_DIR / "diff_spectrograms"
CONTENT_CSV = RESULT_DIR / "sc16_content_features.csv"

SAMPLE_RATE = 16000
DEVICE = "cpu"
MODEL_TYPE = "16k"

# The 15 ternary symbols used during embedding decode to these 3 bytes.
WM_MESSAGE = [24, 97, 134]

MP3_BITRATES = [128, 320]
AAC_BITRATE = 128
RESAMPLE_RATE = 44100
CROP_RATIO = 0.10
TSM_RATE = 1.10
AWGN_SNR_DB = 30
LIMITER_THRESHOLD_DB = -6.0
OPUS_BITRATE = 32
DSR_PASS_RATE = 95.0

# 전체 평가 필수 기능: ViSQOL 점수, EnCodec 공격, DAC 공격.

FEATURE_N_FFT = 1024
FEATURE_HOP_LENGTH = 256
SILENCE_THRESHOLD_DB = -40.0

np.random.seed(42)

try:
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2

    _vq_config = visqol_config_pb2.VisqolConfig()
    _vq_config.audio.sample_rate = SAMPLE_RATE
    _vq_config.options.use_speech_scoring = False
    _vq_config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", "libsvm_nu_svr_model.txt"
    )
    _VISQOL_API = visqol_lib_py.VisqolApi()
    _VISQOL_API.Create(_vq_config)
    VISQOL_AVAILABLE = True
except Exception as error:
    raise RuntimeError("ViSQOL이 필요합니다. `python -m pip install visqol` 후 다시 실행하세요.") from error


def compute_visqol(reference: np.ndarray, degraded: np.ndarray):
    if not VISQOL_AVAILABLE:
        return None
    try:
        length = min(len(reference), len(degraded))
        result = _VISQOL_API.Run(
            reference[:length].astype(np.float64),
            degraded[:length].astype(np.float64),
        )
        return float(result.moslqo)
    except Exception:
        return None


try:
    import torch
    from encodec import EncodecModel
    from encodec.utils import convert_audio as _encodec_convert_audio

    _encodec_model = EncodecModel.encodec_model_24khz()
    _encodec_model.set_target_bandwidth(6.0)
    _encodec_model.eval()
    ENCODEC_AVAILABLE = True
except Exception as error:
    raise RuntimeError("EnCodec이 필요합니다. `python -m pip install encodec` 후 다시 실행하세요.") from error


def apply_encodec(wav: np.ndarray) -> np.ndarray | None:
    try:
        audio = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        audio = _encodec_convert_audio(
            audio, SAMPLE_RATE, _encodec_model.sample_rate, _encodec_model.channels
        ).unsqueeze(0)
        with torch.no_grad():
            encoded = _encodec_model.encode(audio)
            decoded = _encodec_model.decode(encoded)
        output = decoded.squeeze().numpy()
        output = librosa.resample(
            output.astype(np.float32),
            orig_sr=_encodec_model.sample_rate,
            target_sr=SAMPLE_RATE,
        )
        return np.clip(output.astype(np.float32), -1.0, 1.0)
    except Exception:
        return None


try:
    import dac as _dac_pkg
    from audiotools import AudioSignal as _DacAudioSignal

    _dac_model_path = _dac_pkg.utils.download(model_type="16khz")
    _dac_model = _dac_pkg.DAC.load(_dac_model_path)
    _dac_model.eval()
    DAC_AVAILABLE = True
except Exception as error:
    raise RuntimeError("DAC이 필요합니다. `python -m pip install descript-audio-codec` 후 다시 실행하세요.") from error


def apply_dac(wav: np.ndarray) -> np.ndarray | None:
    try:
        signal = _DacAudioSignal(wav, sample_rate=SAMPLE_RATE)
        signal.resample(_dac_model.sample_rate)
        x = _dac_model.preprocess(signal.audio_data, signal.sample_rate)
        with torch.no_grad():
            z, _, _, _, _ = _dac_model.encode(x)
            decoded = _dac_model.decode(z)
        output = decoded.squeeze().cpu().numpy()
        output = librosa.resample(
            output.astype(np.float32),
            orig_sr=_dac_model.sample_rate,
            target_sr=SAMPLE_RATE,
        )
        return np.clip(output.astype(np.float32), -1.0, 1.0)
    except Exception:
        return None


def load_wav(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.clip(y.astype(np.float32), -1.0, 1.0)


def extract_content_features(audio: np.ndarray, sr: int) -> dict:
    """원본 오디오의 장르/콘텐츠 차이 분석용 5개 지표를 계산한다."""
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        return {
            "SpectralFlatness": "", "SpectralCentroid_Hz": "",
            "EffectiveBandwidth99_Hz": "", "SilenceRatio": "",
            "CrestFactor": "",
        }

    stft = librosa.stft(audio, n_fft=FEATURE_N_FFT,
                        hop_length=FEATURE_HOP_LENGTH, center=True)
    magnitude = np.abs(stft)
    power = magnitude ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FEATURE_N_FFT)

    flatness = librosa.feature.spectral_flatness(S=magnitude)
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
    cumulative_power = np.cumsum(np.sum(power, axis=1))
    if cumulative_power[-1] > 0:
        bandwidth_99 = freqs[np.searchsorted(
            cumulative_power, cumulative_power[-1] * 0.99)]
    else:
        bandwidth_99 = 0.0

    rms = librosa.feature.rms(
        y=audio, frame_length=FEATURE_N_FFT,
        hop_length=FEATURE_HOP_LENGTH, center=True,
    )[0]
    max_rms = float(np.max(rms)) if len(rms) else 0.0
    if max_rms <= 1e-12:
        silence_ratio = 1.0
    else:
        relative_db = 20 * np.log10(np.maximum(rms, 1e-12) / max_rms)
        silence_ratio = float(np.mean(relative_db <= SILENCE_THRESHOLD_DB))

    rms_all = float(np.sqrt(np.mean(audio ** 2)))
    crest_factor = float(np.max(np.abs(audio)) / max(rms_all, 1e-12))
    return {
        "SpectralFlatness": float(np.mean(flatness)),
        "SpectralCentroid_Hz": float(np.mean(centroid)),
        "EffectiveBandwidth99_Hz": float(bandwidth_99),
        "SilenceRatio": silence_ratio,
        "CrestFactor": crest_factor,
    }


def apply_mp3(wav: np.ndarray, kbps: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        sf.write(str(temp / "in.wav"), wav, SAMPLE_RATE, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-i", str(temp / "in.wav"),
                        "-b:a", f"{kbps}k", str(temp / "out.mp3")],
                       capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(temp / "out.mp3"),
                        "-ar", str(SAMPLE_RATE), str(temp / "out.wav")],
                       capture_output=True, check=True)
        out, _ = librosa.load(str(temp / "out.wav"), sr=SAMPLE_RATE, mono=True)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def apply_opus(wav: np.ndarray, kbps: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        sf.write(str(temp / "in.wav"), wav, SAMPLE_RATE, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-i", str(temp / "in.wav"),
                        "-c:a", "libopus", "-b:a", f"{kbps}k",
                        str(temp / "out.opus")], capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(temp / "out.opus"),
                        "-ar", str(SAMPLE_RATE), str(temp / "out.wav")],
                       capture_output=True, check=True)
        out, _ = librosa.load(str(temp / "out.wav"), sr=SAMPLE_RATE, mono=True)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def apply_aac(wav: np.ndarray, kbps: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        sf.write(str(temp / "in.wav"), wav, SAMPLE_RATE, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-i", str(temp / "in.wav"),
                        "-c:a", "aac", "-b:a", f"{kbps}k",
                        str(temp / "out.m4a")], capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(temp / "out.m4a"),
                        "-ar", str(SAMPLE_RATE), str(temp / "out.wav")],
                       capture_output=True, check=True)
        out, _ = librosa.load(str(temp / "out.wav"), sr=SAMPLE_RATE, mono=True)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def apply_telephone(wav: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        sf.write(str(temp / "in.wav"), wav, SAMPLE_RATE, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-i", str(temp / "in.wav"), "-af",
                        "highpass=f=300,lowpass=f=3400", "-ar", "8000",
                        "-c:a", "pcm_mulaw", str(temp / "telephone.wav")],
                       capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(temp / "telephone.wav"),
                        "-ar", str(SAMPLE_RATE), str(temp / "out.wav")],
                       capture_output=True, check=True)
        out, _ = librosa.load(str(temp / "out.wav"), sr=SAMPLE_RATE, mono=True)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def apply_resample(wav: np.ndarray) -> np.ndarray:
    up = librosa.resample(wav, orig_sr=SAMPLE_RATE, target_sr=RESAMPLE_RATE)
    down = librosa.resample(up, orig_sr=RESAMPLE_RATE, target_sr=SAMPLE_RATE)
    return np.clip(down.astype(np.float32), -1.0, 1.0)


def apply_crop(wav: np.ndarray) -> np.ndarray:
    cut = int(len(wav) * CROP_RATIO)
    start = np.random.randint(0, max(1, len(wav) - cut))
    return np.concatenate([wav[:start], wav[start + cut:]]).astype(np.float32)


def apply_tsm(wav: np.ndarray) -> np.ndarray:
    return librosa.effects.time_stretch(wav, rate=TSM_RATE).astype(np.float32)


def apply_awgn(wav: np.ndarray) -> np.ndarray:
    signal_power = np.mean(wav ** 2)
    noise_power = signal_power / (10 ** (AWGN_SNR_DB / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(wav))
    return np.clip((wav + noise).astype(np.float32), -1.0, 1.0)


def apply_limiter(wav: np.ndarray) -> np.ndarray:
    threshold = 10 ** (LIMITER_THRESHOLD_DB / 20)
    attack = np.exp(-1.0 / (SAMPLE_RATE * 5 / 1000))
    release = np.exp(-1.0 / (SAMPLE_RATE * 50 / 1000))
    envelope = np.zeros_like(wav)
    current = 0.0
    for index, value in enumerate(np.abs(wav)):
        coefficient = attack if value > current else release
        current = coefficient * current + (1 - coefficient) * value
        envelope[index] = current
    gain = np.where(envelope > threshold, threshold / (envelope + 1e-9), 1.0)
    return np.clip((wav * gain).astype(np.float32), -1.0, 1.0)


def compute_ber(original: list, decoded: list) -> float:
    if decoded is None or len(decoded) != len(original):
        return 1.0
    original_bits = "".join(f"{value:08b}" for value in original)
    decoded_bits = "".join(f"{value:08b}" for value in decoded)
    errors = sum(a != b for a, b in zip(original_bits, decoded_bits))
    return errors / len(original_bits)


def decode_one(model, attacked: np.ndarray) -> dict:
    result = model.decode_wav(attacked, SAMPLE_RATE, phase_shift_decoding=False)
    decoded = result["messages"][0] if result["messages"] else None
    confidence = result["confidences"][0] if result["confidences"] else 0.0
    return {
        "detected": bool(result["status"]),
        "ber": compute_ber(WM_MESSAGE, decoded),
        "confidence": float(confidence),
    }


def get_stoi(reference: np.ndarray, degraded: np.ndarray) -> float:
    try:
        length = min(len(reference), len(degraded))
        return float(calc_stoi(reference[:length], degraded[:length],
                               SAMPLE_RATE, extended=False))
    except Exception:
        return float("nan")


def si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    length = min(len(reference), len(estimate))
    reference = reference[:length] - np.mean(reference[:length])
    estimate = estimate[:length] - np.mean(estimate[:length])
    target = (np.sum(estimate * reference) /
              (np.sum(reference ** 2) + 1e-8)) * reference
    noise = estimate - target
    return float(10 * np.log10((np.sum(target ** 2) + 1e-8) /
                               (np.sum(noise ** 2) + 1e-8)))


def save_diff_spectrogram(reference: np.ndarray, degraded: np.ndarray,
                          output_path: Path) -> None:
    length = min(len(reference), len(degraded))
    reference, degraded = reference[:length], degraded[:length]
    hop = 256
    reference_spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(reference, n_fft=1024, hop_length=hop)), ref=np.max)
    degraded_spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(degraded, n_fft=1024, hop_length=hop)), ref=np.max)
    difference = degraded_spec - reference_spec

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, data, title, cmap in zip(
        axes,
        [reference_spec, degraded_spec, difference],
        ["원본", "워터마크/공격 결과", "차이 (결과 - 원본)"],
        ["magma", "magma", "coolwarm"],
    ):
        image = librosa.display.specshow(data, sr=SAMPLE_RATE, hop_length=hop,
                                         x_axis="time", y_axis="hz",
                                         ax=axis, cmap=cmap)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, format="%+2.0f dB")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(figure)


def attack_list():
    attacks = [(f"MP3_{bitrate}kbps", lambda wav, b=bitrate: apply_mp3(wav, b), False)
               for bitrate in MP3_BITRATES]
    attacks.extend([
        (f"Resample_{RESAMPLE_RATE}RT", apply_resample, False),
        (f"Crop_{int(CROP_RATIO * 100)}pct", apply_crop, True),
        (f"TSM_{TSM_RATE}x", apply_tsm, True),
        (f"AWGN_{AWGN_SNR_DB}dB", apply_awgn, False),
        ("Limiter", apply_limiter, False),
        (f"Opus_{OPUS_BITRATE}kbps", lambda wav: apply_opus(wav, OPUS_BITRATE), False),
        (f"AAC_{AAC_BITRATE}kbps", lambda wav: apply_aac(wav, AAC_BITRATE), False),
        ("TelephoneFilter", apply_telephone, False),
    ])
    attacks.append(("EnCodec_6kbps", apply_encodec, False))
    attacks.append(("DAC", apply_dac, False))
    return attacks


def main():
    if not EMBED_DIR.exists():
        raise FileNotFoundError(f"embed 폴더가 없습니다: {EMBED_DIR}")
    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(f"original 폴더가 없습니다: {ORIGINAL_DIR}")
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    embed_files = sorted(EMBED_DIR.rglob("*.wav"))
    if not embed_files:
        raise RuntimeError(f"embed 폴더에 WAV 파일이 없습니다: {EMBED_DIR}")

    print(f"SC-16 모델 로딩 중... ({MODEL_TYPE})")
    model = silentcipher.get_model(model_type=MODEL_TYPE, device=DEVICE)
    print(f"처리 대상: embed WAV {len(embed_files)}개")

    stages = [("NoAttack", lambda wav: wav, False)] + attack_list()
    rows = []
    content_rows = []
    for index, embed_path in enumerate(embed_files, 1):
        relative = embed_path.relative_to(EMBED_DIR)
        original_name = relative.name[3:] if relative.name.startswith("wm_") else relative.name
        original_path = ORIGINAL_DIR / relative.parent / original_name
        if not original_path.exists():
            print(f"[{index}] 원본 없음, 건너뜀: {relative}")
            continue

        embedded = load_wav(embed_path)
        original = load_wav(original_path)
        content_features = extract_content_features(original, SAMPLE_RATE)
        content_rows.append({"file": str(relative), **content_features})
        for stage, attack, desync in stages:
            try:
                attacked = attack(embedded)
                scores = decode_one(model, attacked)
                length = min(len(original), len(attacked))
                diff_path = SPEC_DIR / relative.parent / f"{relative.stem}_{stage}_diff.png"
                save_diff_spectrogram(original[:length], attacked[:length], diff_path)
                visqol_score = "" if desync else compute_visqol(original, attacked)
                row = {
                    "file": str(relative),
                    "stage": stage,
                    "BER": round(scores["ber"], 6),
                    "detected": int(scores["detected"]),
                    "confidence": round(scores["confidence"], 6),
                    "SI-SNR(dB)": "" if desync else round(si_snr(original, attacked), 4),
                    "STOI": "" if desync else round(get_stoi(original, attacked), 6),
                    "ViSQOL": round(visqol_score, 6) if visqol_score is not None and visqol_score != "" else "",
                    "diff_spectrogram": str(diff_path),
                    "SpectralFlatness": round(content_features["SpectralFlatness"], 6),
                    "SpectralCentroid_Hz": round(content_features["SpectralCentroid_Hz"], 2),
                    "EffectiveBandwidth99_Hz": round(content_features["EffectiveBandwidth99_Hz"], 2),
                    "SilenceRatio": round(content_features["SilenceRatio"], 6),
                    "CrestFactor": round(content_features["CrestFactor"], 4),
                }
                rows.append(row)
                stoi_display = row["STOI"] if row["STOI"] == "" else f"{row['STOI']:.4f}"
                print(f"[{index}/{len(embed_files)}] {relative} | {stage} | "
                    f"BER={row['BER']:.4f} DSR={row['detected']} "
                    f"STOI={stoi_display}")
            except Exception as error:
                print(f"[{index}] {relative} | {stage} 실패: {error}")

    fields = ["file", "stage", "BER", "detected", "confidence",
              "SI-SNR(dB)", "STOI", "ViSQOL", "diff_spectrogram",
              "SpectralFlatness", "SpectralCentroid_Hz",
              "EffectiveBandwidth99_Hz", "SilenceRatio", "CrestFactor"]
    detail_path = RESULT_DIR / "sc16_pipeline_results.csv"
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    content_fields = ["file", "SpectralFlatness", "SpectralCentroid_Hz",
                      "EffectiveBandwidth99_Hz", "SilenceRatio", "CrestFactor"]
    with open(CONTENT_CSV, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=content_fields)
        writer.writeheader()
        writer.writerows(content_rows)

    summary = []
    for stage in dict.fromkeys(row["stage"] for row in rows):
        stage_rows = [row for row in rows if row["stage"] == stage]
        dsr = np.mean([row["detected"] for row in stage_rows]) * 100
        summary.append({
            "stage": stage,
            "files": len(stage_rows),
            "mean_BER": round(float(np.mean([row["BER"] for row in stage_rows])), 6),
            "DSR(%)": round(float(dsr), 2),
            "mean_SI-SNR(dB)": round(float(np.mean([row["SI-SNR(dB)"] for row in stage_rows
                                                     if row["SI-SNR(dB)"] != ""])), 4)
            if any(row["SI-SNR(dB)"] != "" for row in stage_rows) else "",
            "mean_STOI": round(float(np.nanmean([row["STOI"] for row in stage_rows
                                                  if row["STOI"] != ""])), 6)
            if any(row["STOI"] != "" for row in stage_rows) else "",
            "mean_ViSQOL": round(float(np.mean([row["ViSQOL"] for row in stage_rows
                                                   if row["ViSQOL"] != ""])), 6)
            if any(row["ViSQOL"] != "" for row in stage_rows) else "",
            "status": "PASS" if dsr >= DSR_PASS_RATE else "FAIL",
        })
    summary_path = RESULT_DIR / "sc16_pipeline_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()) if summary else [])
        writer.writeheader()
        writer.writerows(summary)

    print(f"상세 결과: {detail_path}")
    print(f"요약 결과: {summary_path}")
    print(f"콘텐츠 특성: {CONTENT_CSV}")
    print(f"차이 스펙트로그램: {SPEC_DIR}")


if __name__ == "__main__":
    main()
