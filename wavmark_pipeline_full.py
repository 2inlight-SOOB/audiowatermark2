r"""
실행 전 환경 확인
------------------
# tmux 세션 방식 (추천 — 나중에 다시 접속해서 진행 상황 볼 수 있음)
tmux new -s wm
python3 /home/hansung1/.vscode-server/audio/wavmark_pipeline_full.py
# Ctrl+B, D 로 detach 하고 노트북 닫아도 됨
# 나중에: tmux attach -t wm

# 또는 nohup 방식
nohup python3 /home/hansung1/.vscode-server/audio/wavmark_pipeline_full.py > pipeline.log 2>&1 &
disown

VS Code에서 `Ctrl + Shift + P` → `Python: Select Interpreter`를 실행한 뒤
기존에 쓰던 가상환경(`/home/hansung1/.vscode-server/audio/venv`)을 선택하세요.
MP3/AAC/전화망 공격에는 시스템 `ffmpeg`도 필요합니다.

참고: WavMark는 `audioseal`을 사용하지 않으므로 AudioSeal 진단은 발생하지 않습니다.
다음 패키지는 전체 평가에 필요합니다.
- `visqol`: 기존 공격 결과의 음질 점수(ViSQOL) 계산
- `encodec`: EnCodec 신경망 코덱 공격 추가
- `descript-audio-codec`: DAC 신경망 코덱 공격 추가
세 패키지를 모두 설치해야 ViSQOL, EnCodec, DAC까지 포함한 전체 평가가 실행됩니다.

WavMark 통합 파이프라인 — 임베딩 -> 무공격 지표 -> 축1/축2/축3 공격 지표 (원스톱)
====================================================================
as_pipeline_full.py(AudioSeal), sc44_pipeline_full.py/sc16_pipeline_full.py(SilentCipher)와
동일한 구조. 비교 기준은 항상 "원본"(mono+16kHz 전처리본)이며, WavMark 특성(1.1초
청크마다 1초 워터마크 + 0.1초 sync 버퍼, 슬라이딩 윈도우 디코딩)에 맞춰
MIoU/F1/IoU(위치 탐지) 지표를 추가로 구현했다.

STEP 1. original/ 의 원본을 mono+16kHz로 변환해 WavMark 워터마크 삽입
STEP 2. 무공격 지표: BER, NC, DSR, MIoU·F1·IoU, SI-SNR, ViSQOL, ODG, 차이 스펙트로그램
    + 콘텐츠 특성 5개: Spectral Flatness, Spectral Centroid, Effective Bandwidth(99%), Silence Ratio, Crest Factor
STEP 3. 3개 축의 공격을 적용한 뒤 동일한 지표 세트를 재계산
    (as_pipeline_full.py/sc44_pipeline_full.py/sc16_pipeline_full.py와 동일 원칙: desync가
     아닌 공격은 축과 무관하게 전부 BER/NC/DSR, SI-SNR, ViSQOL, ODG, MIoU/F1/IoU를 계산한다)
    축1 (AWGN, 리샘플링, 리미터)              : BER/NC/DSR, SI-SNR, ViSQOL, ODG, 차이 스펙트로그램, MIoU/F1/IoU
    축2 (MP3/AAC, 전화망 필터, Crop, TSM)      : BER/NC/DSR, SI-SNR, ViSQOL, ODG, MIoU/F1/IoU
                                                 - Crop/TSM만 시간축이 어긋나 SI-SNR/ViSQOL/ODG 생략, MIoU만 계산
    축3 (EnCodec, DAC 뉴럴 코덱)               : BER/NC/DSR, SI-SNR, ViSQOL, ODG, MIoU/F1/IoU
                                                 - Bit Accuracy(=1-BER)는 WavMark 전용 보너스 지표로 추가 기록
                                                 - MUSHRA는 주관 청취평가라 자동 계산 불가 -> 스티뮬러스 파일만 저장(보너스)

WavMark 디코딩은 전체 신호를 슬라이딩 윈도우로 훑는 방식이라 AudioSeal보다 훨씬
느리다 (긴 트랙 하나당 공격 조건별로 수백~수천 번의 배치 추론 발생). 전체 실행에
시간이 오래 걸릴 수 있다.

결과: wavmark_pipeline_results/pipeline_full_results.csv  (파일 x 단계 전체 결과)
      wavmark_pipeline_results/pipeline_full_summary.csv  (단계별 평균 요약)
    wavmark_pipeline_results/content_features.csv        (원본 파일별 콘텐츠 특성)
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # GPU 0번만 사용

import csv
import time
import hashlib
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt

import wavmark
from wavmark.utils import wm_add_util

matplotlib.rcParams["font.family"] = "Baekmuk Gulim"   # 한글 깨짐 방지 (Linux)
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
#  설정
# ============================================================
AUDIO_ROOT = Path("/home/hansung1/.vscode-server/audio")
INPUT_DIR  = AUDIO_ROOT / "original"

# 테스트용: 각 파일을 앞부분 N초만 잘라서 처리 (WavMark 디코딩이 느려서 전체 길이는
# 오래 걸림). None이면 전체 길이 그대로 처리. 이미 N초보다 짧은 파일은 그대로 둠.
# 이 값이 설정되면 wavmark_new_embed.py가 만든 전체 길이 embed 결과를 덮어쓰지
# 않도록 별도 폴더(_clip{N}s)에 저장한다.
TEST_CLIP_SECONDS = 9
_SUFFIX = "" if TEST_CLIP_SECONDS is None else f"_clip{TEST_CLIP_SECONDS}s"

EMBED_DIR   = AUDIO_ROOT / f"wavmark_embed{_SUFFIX}"
# wavmark_new_embed.py가 이미 만들어 둔 전체 길이 임베딩 결과 (있으면 재사용해서
# 느린 wavmark.encode_watermark 호출을 생략한다 — 파일명은 그대로 wavmark_embed/)
SOURCE_EMBED_DIR = AUDIO_ROOT / "wavmark_embed"
RESULT_DIR  = AUDIO_ROOT / f"wavmark_pipeline_results{_SUFFIX}"
ORIG16K_DIR = RESULT_DIR / "original_16k_mono"
SPEC_DIR    = RESULT_DIR / "diff_spectrograms"
MUSHRA_DIR  = RESULT_DIR / "mushra_stimuli"
CONTENT_CSV = RESULT_DIR / "content_features.csv"

SAMPLE_RATE      = 16000
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
WM_ID            = "2inlightSOOB-0001"
PATTERN_BIT_LEN  = 16
NUM_POINT        = 16000     # 1개 워터마크 청크 = 1초
SHIFT_RANGE      = 0.1       # 청크당 0.1초(1600 샘플) sync 버퍼
SHIFT_RANGE_P    = 0.5       # 디코딩 슬라이딩 스텝 비율
CHUNK_SIZE       = NUM_POINT + int(NUM_POINT * SHIFT_RANGE)          # 17600
DECODE_SHIFT_STEP = int(SHIFT_RANGE * NUM_POINT * SHIFT_RANGE_P)     # 800

# GPU 온도 제한 (nvidia-smi 기반)
GPU_INDEX      = 0
TEMP_LIMIT_C   = 78.0   # 이 온도 이상이면 일시정지
TEMP_RESUME_C  = 70.0   # 이 온도 이하로 내려가야 재개
TEMP_POLL_SEC  = 10

DSR_PASS_RATE = 0.95

SAVE_DIFF_SPECTROGRAM = True   # NoAttack + 모든 공격 단계에서 저장
MAX_SPECTROGRAM_SAVE  = None   # 저장 개수 제한. None이면 무제한

# 공격 파라미터
MP3_BITRATES          = [128, 320]
AAC_BITRATE_KBPS      = 128
RESAMPLE_ROUNDTRIP_SR = 44100
CROP_RATIO            = 0.10
TSM_RATE              = 1.10
AWGN_SNR_DB           = 30
LIMITER_THRESHOLD_DB  = -6.0
OPUS_BITRATE_KBPS     = 32

FEATURE_N_FFT          = 1024
FEATURE_HOP_LENGTH     = 256
SILENCE_THRESHOLD_DB   = -40.0

np.random.seed(42)

# ============================================================
#  ViSQOL / ODG (선택적 — 설치돼 있을 때만 계산, 없으면 자동 스킵)
#  ViSQOL의 audio(일반 음질) 모드는 48kHz 기준으로 학습된 SVR 모델이라 sc44_pipeline_full.py와
#  동일하게 내부적으로 48kHz로 리샘플링해서 호출한다 (16kHz 그대로 넣으면 검증 안 된 구간에서
#  모델을 돌리는 셈이라 MOS-LQO 수치의 신뢰도가 떨어짐).
# ============================================================
VISQOL_SR = 48000
try:
    from visqol.api import VisqolApi   # pip install "visqol-python[all]" (PyPI 순정 배포판)

    _VISQOL_API = VisqolApi()
    _VISQOL_API.create(mode="audio")
    VISQOL_AVAILABLE = True
except Exception as e:
    raise RuntimeError("ViSQOL이 필요합니다. `python -m pip install \"visqol-python[all]\"` 후 다시 실행하세요.") from e


def compute_visqol(ref, deg, sr):
    if not VISQOL_AVAILABLE:
        return None
    try:
        if sr != VISQOL_SR:
            ref = librosa.resample(ref, orig_sr=sr, target_sr=VISQOL_SR)
            deg = librosa.resample(deg, orig_sr=sr, target_sr=VISQOL_SR)
        result = _VISQOL_API.measure_from_arrays(ref.astype(np.float64), deg.astype(np.float64), VISQOL_SR)
        return result.moslqo
    except Exception as e:
        print(f"    ViSQOL 계산 실패: {e}")
        return None


def compute_odg(ref, deg, sr):
    """PEAQ 기반 ODG. 유지보수되는 파이썬 패키지가 없어 기본 비활성화."""
    return None


# ------------------------------------------------------------
#  EnCodec / DAC (선택적 — 설치돼 있으면 축3 공격에 자동 추가)
# ------------------------------------------------------------
try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio as _encodec_convert_audio

    _encodec_model = EncodecModel.encodec_model_24khz().to(DEVICE)
    _encodec_model.set_target_bandwidth(6.0)
    _encodec_model.eval()
    ENCODEC_AVAILABLE = True
except Exception as e:
    ENCODEC_AVAILABLE = False
    raise RuntimeError("EnCodec이 필요합니다. `python -m pip install encodec` 후 다시 실행하세요.") from e


def attack_encodec(wm, sr):
    if not ENCODEC_AVAILABLE:
        return None
    try:
        wav = torch.tensor(wm, dtype=torch.float32).unsqueeze(0)
        wav = _encodec_convert_audio(wav, sr, _encodec_model.sample_rate, _encodec_model.channels)
        wav = wav.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            encoded = _encodec_model.encode(wav)
            decoded = _encodec_model.decode(encoded)
        decoded = decoded.squeeze().cpu().numpy()
        decoded = librosa.resample(decoded.astype(np.float32), orig_sr=_encodec_model.sample_rate, target_sr=sr)
        return np.clip(decoded.astype(np.float32), -1.0, 1.0)
    except Exception as e:
        print(f"    EnCodec 공격 실패: {e}")
        return None


try:
    import dac as _dac_pkg
    from audiotools import AudioSignal as _DacAudioSignal

    _dac_model_path = _dac_pkg.utils.download(model_type="16khz")
    _dac_model = _dac_pkg.DAC.load(_dac_model_path).to(DEVICE)
    _dac_model.eval()
    DAC_AVAILABLE = True
except Exception:
    raise RuntimeError("DAC이 필요합니다. `python -m pip install descript-audio-codec` 후 다시 실행하세요.")


def attack_dac(wm, sr):
    if not DAC_AVAILABLE:
        return None
    try:
        signal = _DacAudioSignal(wm, sample_rate=sr)
        signal.resample(_dac_model.sample_rate)
        x = _dac_model.preprocess(signal.audio_data, signal.sample_rate).to(DEVICE)
        with torch.no_grad():
            z, codes, latents, _, _ = _dac_model.encode(x)
            y = _dac_model.decode(z)
        out = y.squeeze().cpu().numpy()
        out = librosa.resample(out.astype(np.float32), orig_sr=_dac_model.sample_rate, target_sr=sr)
        return np.clip(out.astype(np.float32), -1.0, 1.0)
    except Exception as e:
        print(f"    DAC 공격 실패: {e}")
        return None


# ============================================================
#  GPU 온도 감시
# ============================================================
def get_gpu_temp(gpu_index=GPU_INDEX):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_index}",
             "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def wait_for_safe_temp(limit_c=TEMP_LIMIT_C, resume_c=TEMP_RESUME_C,
                        gpu_index=GPU_INDEX, poll_sec=TEMP_POLL_SEC):
    """GPU 온도가 limit_c 이상이면 resume_c 이하로 내려갈 때까지 대기."""
    temp = get_gpu_temp(gpu_index)
    if temp is None or temp < limit_c:
        return
    print(f"  [THERMAL] GPU{gpu_index} {temp:.0f}C >= {limit_c:.0f}C -> 일시정지, "
          f"{resume_c:.0f}C 이하로 내려가면 재개")
    while True:
        time.sleep(poll_sec)
        temp = get_gpu_temp(gpu_index)
        if temp is None or temp <= resume_c:
            print(f"  [THERMAL] 재개 (현재 {temp}C)" if temp is not None else "  [THERMAL] 온도 조회 실패, 재개")
            return
        print(f"  [THERMAL] 대기 중... 현재 {temp:.0f}C")


# ============================================================
#  오디오 로드 / 전처리
# ============================================================
def load_audio_raw(path):
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def to_model_rate(audio, sr):
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


def extract_content_features(audio, sr):
    """원본 오디오에서 장르/콘텐츠 차이 분석용 5개 지표를 계산한다."""
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


def make_payload(wm_id: str, n_bits: int = 16) -> np.ndarray:
    h = hashlib.sha256(wm_id.encode("utf-8")).digest()
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8)).astype(np.uint8)
    return bits[:n_bits]


# ============================================================
#  공통 지표 함수
# ============================================================
def si_snr(reference, estimate, eps=1e-8):
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    s_target = (np.sum(estimate * reference) / (np.sum(reference ** 2) + eps)) * reference
    e_noise = estimate - s_target
    ratio = (np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps)
    return 10 * np.log10(ratio + eps)


def normalized_correlation(bits_a, bits_b):
    # dtype을 먼저 float64로 바꾼 뒤 연산해야 함 (uint8 상태로 *2-1 하면 0비트에서 언더플로우 발생)
    a = bits_a.astype(np.float64) * 2 - 1
    b = bits_b.astype(np.float64) * 2 - 1
    return float(np.sum(a * b) / len(a))


def save_diff_spectrogram(orig, wm, sr, out_path):
    n_fft, hop = 1024, 256
    S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig, n_fft=n_fft, hop_length=hop)), ref=np.max)
    S_wm = librosa.amplitude_to_db(np.abs(librosa.stft(wm, n_fft=n_fft, hop_length=hop)), ref=np.max)
    S_diff = S_wm - S_orig

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, S, title, cmap in zip(
        axes, [S_orig, S_wm, S_diff],
        ["원본(16k)", "워터마크 삽입/공격", "차이 (대상 - 원본)"],
        ["magma", "magma", "coolwarm"],
    ):
        img = librosa.display.specshow(S, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=ax, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
#  WavMark 디코딩 / BER / NC / DSR
# ============================================================
def total_windows(n_samples):
    return max(1, (n_samples - NUM_POINT) // DECODE_SHIFT_STEP)


def decode_and_measure(model, audio, payload_bits):
    dec_payload, dec_info = wavmark.decode_watermark(
        model, audio, len_start_bit=PATTERN_BIT_LEN, show_progress=False
    )
    results = dec_info.get("results", []) if isinstance(dec_info, dict) else []
    detected = dec_payload is not None

    if detected:
        decoded_bits = np.array(dec_payload, dtype=np.int32)
        ber = float(np.mean(decoded_bits != payload_bits))
        nc = normalized_correlation(payload_bits, decoded_bits)
    else:
        ber = 1.0
        nc = -1.0

    dsr = 1 if detected else 0
    hit_ratio = len(results) / total_windows(len(audio))
    return ber, nc, dsr, hit_ratio, results


# ============================================================
#  MIoU / F1 / IoU (위치 탐지)
# ============================================================
def build_gt_mask(n_samples):
    """임베딩 청크 구조(1.1초 = 1초 워터마크 + 0.1초 버퍼) 기준 정답 마스크."""
    mask = np.zeros(n_samples, dtype=np.uint8)
    n_seg = n_samples // CHUNK_SIZE
    for i in range(n_seg):
        s = i * CHUNK_SIZE
        mask[s:s + NUM_POINT] = 1
    return mask


def build_pred_mask(n_samples, results):
    mask = np.zeros(n_samples, dtype=np.uint8)
    for r in results:
        p = r["start_position"]
        mask[p:min(p + NUM_POINT, n_samples)] = 1
    return mask


def localization_metrics(gt_mask, pred_mask, eps=1e-9):
    n = min(len(gt_mask), len(pred_mask))
    gt = gt_mask[:n].astype(bool)
    pr = pred_mask[:n].astype(bool)
    tp = np.sum(gt & pr)
    fp = np.sum(~gt & pr)
    fn = np.sum(gt & ~pr)
    tn = np.sum(~gt & ~pr)
    iou_wm = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    miou = (iou_wm + iou_bg) / 2
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {"IoU": float(iou_wm), "MIoU": float(miou), "F1": float(f1)}


# ============================================================
#  공격 함수 (16kHz mono 입력 -> 16kHz mono 출력)
# ============================================================
def attack_mp3(wm, sr, bitrate_kbps):
    with tempfile.TemporaryDirectory() as td:
        wav_in, mp3, wav_out = Path(td) / "in.wav", Path(td) / "c.mp3", Path(td) / "out.wav"
        sf.write(str(wav_in), wm, sr)
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_in), "-b:a", f"{bitrate_kbps}k", str(mp3)],
                        capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(mp3), "-ar", str(sr), str(wav_out)],
                        capture_output=True)
        out, _ = sf.read(str(wav_out), always_2d=False)
        if out.ndim == 2:
            out = out.mean(axis=1)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def attack_aac(wm, sr, bitrate_kbps):
    with tempfile.TemporaryDirectory() as td:
        wav_in, m4a, wav_out = Path(td) / "in.wav", Path(td) / "c.m4a", Path(td) / "out.wav"
        sf.write(str(wav_in), wm, sr)
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_in), "-c:a", "aac", "-b:a", f"{bitrate_kbps}k", str(m4a)],
                        capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(m4a), "-ar", str(sr), str(wav_out)],
                        capture_output=True)
        out, _ = sf.read(str(wav_out), always_2d=False)
        if out.ndim == 2:
            out = out.mean(axis=1)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def attack_telephone(wm, sr):
    """전화망 시뮬레이션: 300-3400Hz 대역 제한 + 8kHz mu-law(G.711) 왕복."""
    with tempfile.TemporaryDirectory() as td:
        wav_in, mulaw, wav_out = Path(td) / "in.wav", Path(td) / "c.wav", Path(td) / "out.wav"
        sf.write(str(wav_in), wm, sr)
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_in),
                         "-af", "highpass=f=300,lowpass=f=3400",
                         "-ar", "8000", "-c:a", "pcm_mulaw", str(mulaw)],
                        capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(mulaw), "-ar", str(sr), str(wav_out)],
                        capture_output=True)
        out, _ = sf.read(str(wav_out), always_2d=False)
        if out.ndim == 2:
            out = out.mean(axis=1)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def attack_opus(wm, sr, bitrate_kbps):
    with tempfile.TemporaryDirectory() as td:
        wav_in, opus, wav_out = Path(td) / "in.wav", Path(td) / "c.opus", Path(td) / "out.wav"
        sf.write(str(wav_in), wm, sr)
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_in), "-c:a", "libopus",
                        "-b:a", f"{bitrate_kbps}k", str(opus)], capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(opus), "-ar", str(sr),
                        str(wav_out)], capture_output=True)
        out, _ = sf.read(str(wav_out), always_2d=False)
        if out.ndim == 2:
            out = out.mean(axis=1)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def attack_resample_roundtrip(wm, sr, mid_sr):
    up = librosa.resample(wm, orig_sr=sr, target_sr=mid_sr)
    down = librosa.resample(up, orig_sr=mid_sr, target_sr=sr)
    return np.clip(down.astype(np.float32), -1.0, 1.0)


def attack_awgn(wm, snr_db):
    signal_power = np.mean(wm ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(wm))
    return np.clip(wm + noise, -1.0, 1.0).astype(np.float32)


def attack_limiter(wm, sr, threshold_db, attack_ms=5, release_ms=50):
    threshold = 10 ** (threshold_db / 20)
    attack_coef = np.exp(-1.0 / (sr * attack_ms / 1000))
    release_coef = np.exp(-1.0 / (sr * release_ms / 1000))
    abs_wm = np.abs(wm)
    env = np.zeros_like(abs_wm)
    e = 0.0
    for i in range(len(wm)):
        v = abs_wm[i]
        coef = attack_coef if v > e else release_coef
        e = coef * e + (1 - coef) * v
        env[i] = e
    gain = np.where(env > threshold, threshold / (env + 1e-9), 1.0)
    return np.clip(wm * gain, -1.0, 1.0).astype(np.float32)


def attack_crop_with_mask(wm, mask, ratio):
    n = len(wm)
    cut_len = int(n * ratio)
    start = np.random.randint(0, max(1, n - cut_len))
    audio_out = np.concatenate([wm[:start], wm[start + cut_len:]]).astype(np.float32)
    mask_out = np.concatenate([mask[:start], mask[start + cut_len:]])
    return audio_out, mask_out


def attack_tsm_with_mask(wm, mask, rate):
    audio_out = librosa.effects.time_stretch(wm, rate=rate).astype(np.float32)
    n_out = len(audio_out)
    idx = np.round(np.linspace(0, len(mask) - 1, n_out)).astype(int)
    mask_out = mask[idx]
    return audio_out, mask_out


def build_attacks():
    attacks = []
    # ---- 축1: 정렬 가능, ODG까지 계산 ----
    attacks.append({"name": f"AWGN_{AWGN_SNR_DB}dB", "axis": "axis1", "desync": False,
                     "fn": lambda wm: attack_awgn(wm, AWGN_SNR_DB)})
    attacks.append({"name": f"Resample_{RESAMPLE_ROUNDTRIP_SR}RT", "axis": "axis1", "desync": False,
                     "fn": lambda wm: attack_resample_roundtrip(wm, SAMPLE_RATE, RESAMPLE_ROUNDTRIP_SR)})
    attacks.append({"name": "Limiter", "axis": "axis1", "desync": False,
                     "fn": lambda wm: attack_limiter(wm, SAMPLE_RATE, LIMITER_THRESHOLD_DB)})
    # ---- 축2: 압축/전화망 + 동기화 파괴, MIoU가 핵심 ----
    for br in MP3_BITRATES:
        attacks.append({"name": f"MP3_{br}kbps", "axis": "axis2", "desync": False,
                         "fn": lambda wm, br=br: attack_mp3(wm, SAMPLE_RATE, br)})
    attacks.append({"name": f"AAC_{AAC_BITRATE_KBPS}kbps", "axis": "axis2", "desync": False,
                     "fn": lambda wm: attack_aac(wm, SAMPLE_RATE, AAC_BITRATE_KBPS)})
    attacks.append({"name": "TelephoneFilter", "axis": "axis2", "desync": False,
                     "fn": lambda wm: attack_telephone(wm, SAMPLE_RATE)})
    attacks.append({"name": f"Opus_{OPUS_BITRATE_KBPS}kbps", "axis": "axis2", "desync": False,
                    "fn": lambda wm: attack_opus(wm, SAMPLE_RATE, OPUS_BITRATE_KBPS)})
    attacks.append({"name": f"Crop_{int(CROP_RATIO*100)}pct", "axis": "axis2", "desync": True, "combined": True,
                     "fn": lambda wm, mask: attack_crop_with_mask(wm, mask, CROP_RATIO)})
    attacks.append({"name": f"TSM_{TSM_RATE}x", "axis": "axis2", "desync": True, "combined": True,
                     "fn": lambda wm, mask: attack_tsm_with_mask(wm, mask, TSM_RATE)})
    # ---- 축3: 뉴럴 코덱, 설치된 경우에만 자동 추가 ----
    attacks.append({"name": "EnCodec_6kbps", "axis": "axis3", "desync": False,
                    "fn": lambda wm: attack_encodec(wm, SAMPLE_RATE)})
    attacks.append({"name": "DAC", "axis": "axis3", "desync": False,
                    "fn": lambda wm: attack_dac(wm, SAMPLE_RATE)})
    return attacks


# ============================================================
#  축(axis)별 지표 포함 정책
#  AudioSeal/SC-44/SC-16과 동일한 원칙: desync(Crop/TSM)가 아닌 공격은 전부
#  동일한 지표 세트(BER/NC/SI-SNR/ViSQOL/ODG/MIoU)를 계산한다. 축3(EnCodec/DAC)도
#  시간축이 안 깨지는 공격이라 BER/NC/SI-SNR을 뺄 이유가 없어서 다른 축과 통일했다.
#  BitAccuracy(=1-BER)와 MUSHRA 스티뮬러스 저장은 WavMark만의 보너스 지표로 유지.
# ============================================================
AXIS_POLICY = {
    "step2": {"ber_nc": True, "bit_accuracy": False, "si_snr": True, "visqol": True,
              "odg": True, "miou": True, "diff_spec": True, "mushra": False},
    "axis1": {"ber_nc": True, "bit_accuracy": False, "si_snr": True, "visqol": True,
              "odg": True, "miou": True, "diff_spec": True, "mushra": False},
    "axis2": {"ber_nc": True, "bit_accuracy": False, "si_snr": True, "visqol": True,
              "odg": True, "miou": True, "diff_spec": True, "mushra": False},
    "axis3": {"ber_nc": True, "bit_accuracy": True, "si_snr": True, "visqol": True,
              "odg": True, "miou": True, "diff_spec": True, "mushra": True},
}


def build_row(rel_name, stage, axis, orig16k, mask_gt, attacked_audio, model,
              payload_bits, desync, content_features, spec_out=None,
              mushra_out_dir=None):
    policy = dict(AXIS_POLICY[axis])
    if desync:
        policy["si_snr"] = False
        policy["visqol"] = False
        policy["diff_spec"] = False

    ber, nc, dsr, hit_ratio, results = decode_and_measure(model, attacked_audio, payload_bits)

    row = {
        "file": rel_name, "stage": stage, "axis": axis,
        "BER": "", "NC": "", "BitAccuracy": "", "DSR": dsr, "hit_ratio": round(hit_ratio, 4),
        "SI-SNR(dB)": "", "ViSQOL": "", "ODG": "",
        "IoU": "", "MIoU": "", "F1": "", "MUSHRA": "",
        "SpectralFlatness": round(content_features["SpectralFlatness"], 6),
        "SpectralCentroid_Hz": round(content_features["SpectralCentroid_Hz"], 2),
        "EffectiveBandwidth99_Hz": round(content_features["EffectiveBandwidth99_Hz"], 2),
        "SilenceRatio": round(content_features["SilenceRatio"], 6),
        "CrestFactor": round(content_features["CrestFactor"], 4),
    }

    if policy["ber_nc"]:
        row["BER"] = round(ber, 4)
        row["NC"] = round(nc, 4)
    if policy["bit_accuracy"]:
        row["BitAccuracy"] = round(1.0 - ber, 4)

    min_len = min(len(orig16k), len(attacked_audio))
    if policy["si_snr"]:
        row["SI-SNR(dB)"] = round(si_snr(orig16k[:min_len], attacked_audio[:min_len]), 2)
    if policy["visqol"]:
        v = compute_visqol(orig16k[:min_len], attacked_audio[:min_len], SAMPLE_RATE)
        row["ViSQOL"] = round(v, 3) if v is not None else ""
    if policy["odg"]:
        o = compute_odg(orig16k[:min_len], attacked_audio[:min_len], SAMPLE_RATE)
        row["ODG"] = round(o, 3) if o is not None else ""
    if spec_out is not None and policy["diff_spec"]:
        save_diff_spectrogram(orig16k[:min_len], attacked_audio[:min_len], SAMPLE_RATE, spec_out)

    if policy["miou"]:
        pred_mask = build_pred_mask(len(attacked_audio), results)
        loc = localization_metrics(mask_gt, pred_mask)
        row["IoU"] = round(loc["IoU"], 4)
        row["MIoU"] = round(loc["MIoU"], 4)
        row["F1"] = round(loc["F1"], 4)

    if policy["mushra"] and mushra_out_dir is not None:
        mushra_out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(mushra_out_dir / f"{stage}.wav"), attacked_audio, SAMPLE_RATE, subtype="PCM_16")
        row["MUSHRA"] = "수동 평가 필요"

    return row


# ============================================================
#  재개(resume) 지원
# ============================================================
def _cast_numeric(value):
    """CSV에서 다시 읽은 문자열 값을 숫자로 복원 (빈 문자열/텍스트는 그대로 둠)."""
    if value == "":
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_existing_results(csv_path, fieldnames):
    """이전 실행이 중간까지 써 둔 결과 CSV를 읽어 재개(resume)에 사용."""
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {k: r.get(k, "") for k in fieldnames}
            for key in fieldnames:
                if key not in ("file", "stage", "axis", "MUSHRA"):
                    row[key] = _cast_numeric(row[key])
            rows.append(row)
    return rows


# ============================================================
#  메인
# ============================================================
def main():
    for d in (EMBED_DIR, RESULT_DIR, ORIG16K_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if SAVE_DIFF_SPECTROGRAM:
        SPEC_DIR.mkdir(parents=True, exist_ok=True)

    ffmpeg_ok = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    if not ffmpeg_ok:
        print("[ERROR] ffmpeg가 설치되어 있지 않습니다! MP3/AAC/전화망 공격을 실행할 수 없습니다.")
        return

    print("WavMark 모델 로딩 중...")
    model = wavmark.load_model().to(DEVICE)
    model.eval()
    print("완료!\n")

    payload = make_payload(WM_ID, n_bits=PATTERN_BIT_LEN)   # 16bit payload
    pattern = np.array(wm_add_util.fix_pattern[0:PATTERN_BIT_LEN], dtype=np.uint8)
    print(f"WM_ID   : {WM_ID}")
    print(f"payload : {payload.tolist()}")

    wav_files = sorted(INPUT_DIR.rglob("*.wav"))
    print(f"\n총 원본 파일 수: {len(wav_files)}개")

    attacks = build_attacks()
    expected_stages = {"NoAttack"} | {a["name"] for a in attacks}
    print(f"공격 단계: {[a['name'] for a in attacks]}")
    print("=" * 60)
    print("※ WavMark 디코딩은 슬라이딩 윈도우 전수 탐색 방식이라 긴 트랙에서는")
    print("   AudioSeal 대비 훨씬 오래 걸릴 수 있습니다.")
    print("=" * 60)

    fieldnames = ["file", "stage", "axis", "BER", "NC", "BitAccuracy", "DSR", "hit_ratio",
                  "SI-SNR(dB)", "ViSQOL", "ODG", "IoU", "MIoU", "F1", "MUSHRA",
                  "SpectralFlatness", "SpectralCentroid_Hz", "EffectiveBandwidth99_Hz",
                  "SilenceRatio", "CrestFactor"]
    csv_path = RESULT_DIR / "pipeline_full_results.csv"

    all_rows = load_existing_results(csv_path, fieldnames)
    stages_by_file = {}
    for r in all_rows:
        stages_by_file.setdefault(r["file"], set()).add(r["stage"])
    completed_files = {f for f, stages in stages_by_file.items() if expected_stages.issubset(stages)}
    if completed_files:
        print(f"[RESUME] 이전 결과 재사용: {len(completed_files)}개 파일은 이미 완료되어 건너뜁니다.")

    spec_saved = len(list(SPEC_DIR.glob("*.png"))) if SAVE_DIFF_SPECTROGRAM else 0

    for i, p in enumerate(wav_files):
        wait_for_safe_temp()
        rel = p.relative_to(INPUT_DIR)
        if str(rel) in completed_files:
            print(f"\n[{i+1}/{len(wav_files)}] {rel}  [SKIP] 이미 완료됨")
            continue

        print(f"\n[{i+1}/{len(wav_files)}] {rel}")
        # 이 파일에 대한 이전 실행의 부분 결과가 있으면 버리고 다시 계산
        all_rows = [r for r in all_rows if r["file"] != str(rel)]
        try:
            # ---------- STEP 1: 전처리 + 임베딩 ----------
            orig16k_path = ORIG16K_DIR / rel.parent / rel.name
            wm_path = EMBED_DIR / rel.parent / f"wm_{rel.name}"
            source_wm_path = SOURCE_EMBED_DIR / rel.parent / f"wm_{rel.name}"

            if orig16k_path.exists() and wm_path.exists():
                # 이 파이프라인이 이전에 이미 만들어 둔 (클립된) 임베딩 재사용
                orig16k, _ = sf.read(str(orig16k_path), always_2d=False)
                wm16, _ = sf.read(str(wm_path), always_2d=False)
                orig16k = orig16k.astype(np.float32)
                wm16 = wm16.astype(np.float32)
                print(f"  [CACHE]  캐시된 임베딩 재사용 ({len(orig16k)/SAMPLE_RATE:.1f}초)")

            elif source_wm_path.exists():
                # wavmark_new_embed.py가 만들어 둔 전체 길이 임베딩을 재사용해서
                # 느린 wavmark.encode_watermark 호출을 생략한다.
                raw, raw_sr = load_audio_raw(p)
                orig16k = to_model_rate(raw, raw_sr)
                wm16, _ = sf.read(str(source_wm_path), always_2d=False)
                wm16 = wm16.astype(np.float32)
                if TEST_CLIP_SECONDS is not None:
                    clip_n = int(SAMPLE_RATE * TEST_CLIP_SECONDS)
                    orig16k = orig16k[:clip_n]
                    wm16 = wm16[:clip_n]

                if len(orig16k) < CHUNK_SIZE:
                    print(f"  [WARN]  길이가 너무 짧아 스킵 (>= {CHUNK_SIZE/SAMPLE_RATE:.1f}초 필요): {rel}")
                    continue

                orig16k_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(orig16k_path), orig16k, SAMPLE_RATE, subtype="PCM_16")
                wm_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(wm_path), wm16, SAMPLE_RATE, subtype="PCM_16")
                print(f"  [REUSE]  wavmark_embed/ 기존 임베딩 재사용 (encode 생략, "
                      f"{len(orig16k)/SAMPLE_RATE:.1f}초)")

            else:
                raw, raw_sr = load_audio_raw(p)
                orig16k = to_model_rate(raw, raw_sr)
                if TEST_CLIP_SECONDS is not None:
                    orig16k = orig16k[:int(SAMPLE_RATE * TEST_CLIP_SECONDS)]

                if len(orig16k) < CHUNK_SIZE:
                    print(f"  [WARN]  길이가 너무 짧아 스킵 (>= {CHUNK_SIZE/SAMPLE_RATE:.1f}초 필요): {rel}")
                    continue

                wm16, info = wavmark.encode_watermark(
                    model, orig16k, payload,
                    pattern_bit_length=PATTERN_BIT_LEN, min_snr=20, max_snr=38, show_progress=False,
                )
                wm16 = np.asarray(wm16).astype(np.float32).squeeze()
                wm16 = np.clip(wm16, -1.0, 1.0)

                orig16k_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(orig16k_path), orig16k, SAMPLE_RATE, subtype="PCM_16")
                wm_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(wm_path), wm16, SAMPLE_RATE, subtype="PCM_16")
                print(f"  [OK] 임베딩 완료 (원본 sr={raw_sr} -> 16kHz, {len(orig16k)/SAMPLE_RATE:.1f}초, "
                      f"encoded={info.get('encoded_sections')} skip={info.get('skip_sections')})")

            gt_mask = build_gt_mask(len(wm16))
            content_features = extract_content_features(orig16k, SAMPLE_RATE)

            # ---------- STEP 2: 무공격 지표 ----------
            spec_out = None
            if SAVE_DIFF_SPECTROGRAM and (MAX_SPECTROGRAM_SAVE is None or spec_saved < MAX_SPECTROGRAM_SAVE):
                spec_out = SPEC_DIR / f"{rel.stem}_NoAttack_diff.png"
                spec_saved += 1
            row = build_row(str(rel), "NoAttack", "step2", orig16k, gt_mask, wm16, model,
                             payload, desync=False, content_features=content_features,
                             spec_out=spec_out)
            all_rows.append(row)
            print(f"  [NoAttack]  BER={row['BER']} NC={row['NC']} DSR={row['DSR']} "
                  f"MIoU={row['MIoU']} SI-SNR={row['SI-SNR(dB)']}dB")

            # ---------- STEP 3: 공격별 지표 ----------
            for atk in attacks:
                wait_for_safe_temp()
                try:
                    if atk.get("combined"):
                        attacked, mask_for_stage = atk["fn"](wm16, gt_mask)
                    else:
                        attacked = atk["fn"](wm16)
                        mask_for_stage = gt_mask
                    if attacked is None:
                        continue

                    spec_out = None
                    if SAVE_DIFF_SPECTROGRAM and \
                       (MAX_SPECTROGRAM_SAVE is None or spec_saved < MAX_SPECTROGRAM_SAVE):
                        spec_out = SPEC_DIR / f"{rel.stem}_{atk['name']}_diff.png"
                        spec_saved += 1

                    mushra_dir = MUSHRA_DIR / rel.stem if atk["axis"] == "axis3" else None

                    row = build_row(str(rel), atk["name"], atk["axis"], orig16k, mask_for_stage,
                                     attacked, model, payload, desync=atk["desync"],
                                     content_features=content_features, spec_out=spec_out,
                                     mushra_out_dir=mushra_dir)
                    all_rows.append(row)
                    print(f"  [{atk['name']}] BER={row['BER']} BitAcc={row['BitAccuracy']} "
                          f"DSR={row['DSR']} MIoU={row['MIoU']} SI-SNR={row['SI-SNR(dB)']}dB")
                except Exception as e:
                    print(f"  [FAIL] [{atk['name']}] 공격 실패: {e}")

        except Exception as e:
            print(f"  [FAIL] 오류: {rel} → {e}")

        # 파일 하나 끝날 때마다 CSV를 다시 써서, 중간에 멈춰도 그때까지 결과는 남고
        # 다음 실행 시 이어서 처리할 수 있도록 한다.
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    # ---------- 원본 콘텐츠 특성 CSV ----------
    content_fields = ["file", "SpectralFlatness", "SpectralCentroid_Hz",
                      "EffectiveBandwidth99_Hz", "SilenceRatio", "CrestFactor"]
    content_rows = []
    seen_files = set()
    for row in all_rows:
        if row["stage"] != "NoAttack" or row["file"] in seen_files:
            continue
        seen_files.add(row["file"])
        content_rows.append({field: row.get(field, "") for field in content_fields})
    with open(CONTENT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=content_fields)
        writer.writeheader()
        writer.writerows(content_rows)

    # ---------- 단계별 요약 ----------
    stages = list(dict.fromkeys(r["stage"] for r in all_rows))
    summary_rows = []
    for stage in stages:
        stage_rows = [r for r in all_rows if r["stage"] == stage]

        def numeric(key):
            return [r[key] for r in stage_rows if isinstance(r[key], (int, float))]

        dsr_vals = numeric("DSR")
        dsr_rate = np.mean(dsr_vals) if dsr_vals else 0.0
        summary_rows.append({
            "stage": stage,
            "axis": stage_rows[0]["axis"] if stage_rows else "",
            "파일수": len(stage_rows),
            "평균BER": round(float(np.mean(numeric("BER"))), 4) if numeric("BER") else "",
            "평균BitAccuracy": round(float(np.mean(numeric("BitAccuracy"))), 4) if numeric("BitAccuracy") else "",
            "평균NC": round(float(np.mean(numeric("NC"))), 4) if numeric("NC") else "",
            "DSR(%)": round(dsr_rate * 100, 1),
            "평균SI-SNR(dB)": round(float(np.mean(numeric("SI-SNR(dB)"))), 2) if numeric("SI-SNR(dB)") else "",
            "평균ViSQOL": round(float(np.mean(numeric("ViSQOL"))), 3) if numeric("ViSQOL") else "",
            "평균MIoU": round(float(np.mean(numeric("MIoU"))), 4) if numeric("MIoU") else "",
            "평균F1": round(float(np.mean(numeric("F1"))), 4) if numeric("F1") else "",
            "판정": "통과" if dsr_rate >= DSR_PASS_RATE else "붕괴",
        })

    summary_path = RESULT_DIR / "pipeline_full_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 60)
    print("  전체 파이프라인 완료")
    print("=" * 60)
    print(f"  {'단계':<20} {'BER':<8} {'DSR':<8} {'MIoU':<8} 판정")
    print(f"  {'-'*55}")
    for s in summary_rows:
        print(f"  {s['stage']:<20} {s['평균BER'] or s['평균BitAccuracy']:<8} {s['DSR(%)']}%{'':<4} "
              f"{s['평균MIoU']:<8} {s['판정']}")
    print("=" * 60)
    print(f"  전체 결과 CSV : {csv_path}")
    print(f"  요약 CSV      : {summary_path}")
    print(f"  콘텐츠 특성 CSV: {CONTENT_CSV}")
    print(f"  임베딩 결과   : {EMBED_DIR}")
    print(f"  16k 원본      : {ORIG16K_DIR}")
    if SAVE_DIFF_SPECTROGRAM:
        print(f"  차이 스펙트로그램({spec_saved}개): {SPEC_DIR}")
    print(f"  MUSHRA 청취용 파일 (수동 평가 필요): {MUSHRA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
