r"""
실행 전 환경 확인
------------------
# tmux 세션 방식 (추천 — 나중에 다시 접속해서 진행 상황 볼 수 있음)
tmux new -s wm
python3 /home/hansung1/.vscode-server/audio/as_pipeline_full.py
# Ctrl+B, D 로 detach 하고 노트북 닫아도 됨
# 나중에: tmux attach -t wm

# 또는 nohup 방식
nohup python3 /home/hansung1/.vscode-server/audio/as_pipeline_full.py > pipeline.log 2>&1 &
disown

PowerShell에서 먼저 실행:

    & "C:\Users\user\OneDrive\Desktop\audioseal venv\Scripts\Activate.ps1"
    python -m pip install --upgrade pip
    python -c "from audioseal import AudioSeal; print('audioseal OK')"

필수 패키지 설치:

    python -m pip install encodec
    python -m pip install visqol
    python -m pip install descript-audio-codec
    python -m pip install descript-audio-codec

`visqol`, `encodec`, `descript-audio-codec`까지 설치해야 전체 공격/지표가 실행됩니다.

- audioseal: 필수 패키지. VS Code에서 AudioSeal을 설치한 가상환경을 선택해야 합니다.
- visqol: 필수 패키지. 음질 점수 계산에 사용합니다.
- encodec: 필수 패키지. EnCodec 공격에 사용합니다.
- descript-audio-codec: 필수 패키지. DAC 공격에 사용합니다.
- descript-audio-codec: 필수 패키지. DAC 공격에 사용합니다.
- convert_audio 바인딩 경고: encodec import 실패로 인해 따라 나오는 2차 경고입니다.

VS Code에서 `Ctrl + Shift + P` → `Python: Select Interpreter`를 실행한 뒤
AudioSeal을 설치한 가상환경을 선택하세요.

AudioSeal 통합 파이프라인 — 임베딩 → 무공격 지표 → 5종 공격 지표 (원스톱)
====================================================================
STEP 1. original/ 의 원본(어떤 sr·길이든)을 mono+16kHz로 변환해 워터마크 삽입
STEP 2. 원본 콘텐츠 특성 5개 계산: Spectral Flatness, Spectral Centroid,
        Effective Bandwidth(99%), Silence Ratio, Crest Factor
STEP 3. 무공격 상태(원본 vs 워터마크) 지표: BER, NC, DSR, MIoU·F1·IoU, SI-SNR, ViSQOL*, PESQ*, ODG*, 차이 스펙트로그램
STEP 4. 아래 5개 공격 카테고리(총 8개 조건)를 적용한 뒤 동일 지표 재계산
    - 압축        : MP3 128kbps / 320kbps
    - 추가 압축/전화망: AAC 128kbps, TelephoneFilter
    - 형식 변환   : 리샘플링 16kHz → 44.1kHz → 16kHz
    - 동기화 파괴 : Cropping(10% 무작위 절삭), TSM(속도 10% 변조)
    - 잡음/왜곡   : AWGN(SNR 30dB), Limiter
    - 차세대 코덱 : Opus(32kbps) [EnCodec은 설치돼 있으면 자동 추가]

MIoU·F1·IoU(위치 탐지)는 AudioSeal 탐지기가 프레임 단위로 내는 워터마크 확률
(result[:,1,:])을 이용해 "어느 구간이 워터마크로 판정됐는지" 예측 마스크를 만들고,
전체 구간이 워터마크되어 있다는 정답 마스크와 비교해 계산한다. 공격 전(NoAttack)에도
계산해서 삽입·탐지 파이프라인 자체의 위치추적 하한선을 먼저 확인한다.

동기화 파괴 공격(Crop, TSM) 이후에는 탐지(BER/NC/DSR/MIoU·F1·IoU)는 실제 공격된
신호 그대로 측정하되, SI-SNR/ViSQOL/PESQ는 알려진 변형 파라미터로 시간축을
재정렬한 신호 쌍으로 계산한다: Crop은 원본에서도 동일 구간을 잘라내 같은 길이로
비교하고, TSM은 공격 신호를 역배속(1/rate)해 원래 길이로 되돌린 뒤 비교한다.
(정답 마스크도 Crop/TSM과 동일하게 시간 변형해서 정합을 맞춘다. 차이 스펙트로그램도
이 정렬된 신호 쌍 기준으로 저장한다.)

* ViSQOL/PESQ는 전체 평가에 포함되므로 각각 `visqol-python`, `pesq` 패키지를
  설치해야 한다 (미설치 시 해당 값만 빈 칸으로 남고 나머지는 정상 진행).
  ODG(PEAQ)는 유지보수되는 파이썬 패키지가 없어 여전히 미구현이며, 대신 PESQ를
  투명성 지표로 사용한다.

결과: as_pipeline_results/pipeline_full_results.csv   (파일 x 단계 전체 결과)
    as_pipeline_results/pipeline_full_summary.csv   (단계별 평균 요약)
    as_pipeline_results/content_features.csv         (원본 파일별 콘텐츠 특성)
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # GPU 0번만 사용
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import csv
import time
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

from audioseal import AudioSeal

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

matplotlib.rcParams["font.family"] = "Baekmuk Gulim"   # 한글 깨짐 방지 (Linux)
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
#  설정
# ============================================================
AUDIO_ROOT  = Path("/home/hansung1/.vscode-server/audio")
INPUT_DIR   = AUDIO_ROOT / "original"
EMBED_DIR   = AUDIO_ROOT / "embed"
RESULT_DIR  = AUDIO_ROOT / "as_pipeline_results"
ORIG16K_DIR = RESULT_DIR / "original_16k_mono"
SPEC_DIR    = RESULT_DIR / "diff_spectrograms"
CONTENT_CSV = RESULT_DIR / "content_features.csv"

SAMPLE_RATE = 16000
FIXED_MESSAGE = torch.tensor(
    [[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]], dtype=torch.float32
)

DETECTION_THRESHOLD = 0.5    # 파일 단위 DSR 판정 임계값 (탐지 프레임 비율)
DSR_PASS_RATE       = 0.95   # 단계별 요약에서 "통과" 판정 기준

SAVE_DIFF_SPECTROGRAM = True   # NoAttack + 정렬 가능한(desync가 아닌) 모든 공격 단계에서 저장
MAX_SPECTROGRAM_SAVE  = None   # 저장 개수 제한. None이면 무제한 (파일수 x 정렬가능 단계수 만큼 생성됨)

# GPU 온도 제한 (nvidia-smi 기반)
GPU_INDEX      = 0
TEMP_LIMIT_C   = 78.0   # 이 온도 이상이면 일시정지
TEMP_RESUME_C  = 70.0   # 이 온도 이하로 내려가야 재개
TEMP_POLL_SEC  = 10

# 공격 파라미터
MP3_BITRATES          = [128, 320]     # kbps
RESAMPLE_ROUNDTRIP_SR = 44100          # 16k -> 44.1k -> 16k
CROP_RATIO            = 0.10           # 10% 무작위 구간 절삭
TSM_RATE              = 1.10           # 속도 10% 변조 (>1: 빨라짐)
AWGN_SNR_DB           = 30
LIMITER_THRESHOLD_DB  = -6.0
OPUS_BITRATE_KBPS     = 32
AAC_BITRATE_KBPS      = 128

# 콘텐츠 특성 추출 파라미터
FEATURE_N_FFT          = 1024
FEATURE_HOP_LENGTH     = 256
SILENCE_THRESHOLD_DB   = -40.0  # 파일 최고 RMS 대비 상대 기준

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
#  PESQ (선택적 — 설치돼 있을 때만 계산, 없으면 자동 스킵)
#  ODG(PEAQ)의 실질적인 대체 지표. wideband 모드는 16kHz 입력을 요구한다.
# ------------------------------------------------------------
try:
    from pesq import pesq as _pesq_fn

    PESQ_AVAILABLE = True
except Exception as e:
    PESQ_AVAILABLE = False
    print(f"PESQ 미사용 (미설치): {e}")


def compute_pesq(ref, deg, sr):
    if not PESQ_AVAILABLE:
        return None
    try:
        if sr == 16000:
            r, d, pesq_sr, mode = ref, deg, 16000, "wb"
        elif sr == 8000:
            r, d, pesq_sr, mode = ref, deg, 8000, "nb"
        else:
            pesq_sr, mode = 16000, "wb"
            r = librosa.resample(ref, orig_sr=sr, target_sr=pesq_sr)
            d = librosa.resample(deg, orig_sr=sr, target_sr=pesq_sr)
        n = min(len(r), len(d))
        return float(_pesq_fn(pesq_sr, r[:n].astype(np.float32), d[:n].astype(np.float32), mode))
    except Exception as e:
        print(f"    PESQ 계산 실패: {e}")
        return None


# ------------------------------------------------------------
#  EnCodec (선택적 — 설치돼 있으면 차세대 코덱 공격에 자동 추가)
# ------------------------------------------------------------
try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio

    _encodec_model = EncodecModel.encodec_model_24khz().to(DEVICE)
    _encodec_model.set_target_bandwidth(6.0)
    _encodec_model.eval()
    ENCODEC_AVAILABLE = True
except Exception as e:
    ENCODEC_AVAILABLE = False
    raise RuntimeError("EnCodec이 필요합니다. `python -m pip install encodec` 후 다시 실행하세요.") from e


def attack_encodec(wm, sr):
    try:
        wav = torch.tensor(wm, dtype=torch.float32).unsqueeze(0)
        wav = convert_audio(wav, sr, _encodec_model.sample_rate, _encodec_model.channels)
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
    try:
        signal = _DacAudioSignal(wm, sample_rate=sr)
        signal.resample(_dac_model.sample_rate)
        x = _dac_model.preprocess(signal.audio_data, signal.sample_rate).to(DEVICE)
        with torch.no_grad():
            z, _, _, _, _ = _dac_model.encode(x)
            decoded = _dac_model.decode(z)
        out = decoded.squeeze().cpu().numpy()
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
    """mono + AudioSeal이 요구하는 16kHz로 변환 (0.2+ 부터 내부 리샘플링 없음)."""
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


# ============================================================
#  원본 콘텐츠 특성
# ============================================================
def extract_content_features(audio, sr):
    """원본 오디오에서 장르/콘텐츠 차이 설명에 사용할 5개 지표를 계산한다."""
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        return {
            "SpectralFlatness": "", "SpectralCentroid_Hz": "",
            "EffectiveBandwidth99_Hz": "", "SilenceRatio": "",
            "CrestFactor": "",
        }

    stft = librosa.stft(
        audio, n_fft=FEATURE_N_FFT, hop_length=FEATURE_HOP_LENGTH,
        center=True,
    )
    magnitude = np.abs(stft)
    power = magnitude ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FEATURE_N_FFT)

    flatness = librosa.feature.spectral_flatness(S=magnitude)
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
    total_power = np.sum(power, axis=1)
    cumulative = np.cumsum(total_power)
    if cumulative[-1] > 0:
        bandwidth_99 = freqs[np.searchsorted(cumulative, cumulative[-1] * 0.99)]
    else:
        bandwidth_99 = 0.0

    rms = librosa.feature.rms(y=audio, frame_length=FEATURE_N_FFT,
                              hop_length=FEATURE_HOP_LENGTH, center=True)[0]
    max_rms = float(np.max(rms)) if len(rms) else 0.0
    if max_rms <= 1e-12:
        silence_ratio = 1.0
    else:
        silence_ratio = float(np.mean(20 * np.log10(np.maximum(rms, 1e-12) / max_rms)
                                      <= SILENCE_THRESHOLD_DB))

    rms_all = float(np.sqrt(np.mean(audio ** 2)))
    crest_factor = float(np.max(np.abs(audio)) / max(rms_all, 1e-12))
    return {
        "SpectralFlatness": float(np.mean(flatness)),
        "SpectralCentroid_Hz": float(np.mean(centroid)),
        "EffectiveBandwidth99_Hz": float(bandwidth_99),
        "SilenceRatio": silence_ratio,
        "CrestFactor": crest_factor,
    }


# ============================================================
#  지표 함수
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
    a = bits_a.astype(np.float64) * 2 - 1   # {0,1} -> {-1,+1}
    b = bits_b.astype(np.float64) * 2 - 1
    return float(np.sum(a * b) / len(a))


def decode_and_measure(detector, audio, sr):
    """BER/NC/DSR과 함께, MIoU 계산용 샘플 단위 예측 마스크도 반환한다."""
    x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        result, message = detector(x, sample_rate=sr)   # result: (1,2,T) softmax, message: (1,16) sigmoid

    frame_pred = (result[0, 1, :] > DETECTION_THRESHOLD)   # (T,) 프레임별 워터마크 판정
    detect_prob = frame_pred.float().mean().item()
    decoded_bits = (message.squeeze(0) > 0.5).int().cpu().numpy()
    original_bits = FIXED_MESSAGE.squeeze(0).numpy().astype(np.int32)

    ber = float(np.mean(decoded_bits != original_bits))
    nc = normalized_correlation(original_bits, decoded_bits)
    dsr = 1 if detect_prob >= DETECTION_THRESHOLD else 0

    frame_pred_np = frame_pred.cpu().numpy().astype(np.uint8)
    pred_mask = upsample_frame_mask(frame_pred_np, len(audio))
    return ber, nc, dsr, detect_prob, pred_mask


# ------------------------------------------------------------
#  MIoU / F1 / IoU (위치 탐지)
# ------------------------------------------------------------
def build_gt_mask(n_samples):
    """AudioSeal은 클립 전체에 연속적으로 워터마크를 삽입하므로 정답 마스크는 전부 1."""
    return np.ones(n_samples, dtype=np.uint8)


def upsample_frame_mask(frame_mask, n_samples):
    """탐지기의 프레임 단위 예측을 샘플 단위로 최근접 보간."""
    n_frames = len(frame_mask)
    if n_frames == 0:
        return np.zeros(n_samples, dtype=np.uint8)
    idx = np.floor(np.linspace(0, n_frames - 1, n_samples)).astype(int)
    return frame_mask[idx]


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


def save_diff_spectrogram(orig, wm, sr, out_path):
    n_fft, hop = 1024, 256
    S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig, n_fft=n_fft, hop_length=hop)), ref=np.max)
    S_wm = librosa.amplitude_to_db(np.abs(librosa.stft(wm, n_fft=n_fft, hop_length=hop)), ref=np.max)
    S_diff = S_wm - S_orig

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, S, title, cmap in zip(
        axes, [S_orig, S_wm, S_diff],
        ["원본(16k)", "워터마크 삽입", "차이 (WM - 원본)"],
        ["magma", "magma", "coolwarm"],
    ):
        img = librosa.display.specshow(S, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=ax, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
#  공격 함수 (모두 16kHz mono 입력 -> 16kHz mono 출력)
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


def attack_opus(wm, sr, bitrate_kbps):
    with tempfile.TemporaryDirectory() as td:
        wav_in, opus, wav_out = Path(td) / "in.wav", Path(td) / "c.opus", Path(td) / "out.wav"
        sf.write(str(wav_in), wm, sr)
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_in), "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", str(opus)],
                        capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(opus), "-ar", str(sr), str(wav_out)],
                        capture_output=True)
        out, _ = sf.read(str(wav_out), always_2d=False)
        if out.ndim == 2:
            out = out.mean(axis=1)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def attack_aac(wm, sr, bitrate_kbps):
    with tempfile.TemporaryDirectory() as td:
        wav_in, m4a, wav_out = Path(td) / "in.wav", Path(td) / "c.m4a", Path(td) / "out.wav"
        sf.write(str(wav_in), wm, sr)
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_in), "-c:a", "aac",
                        "-b:a", f"{bitrate_kbps}k", str(m4a)], capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(m4a), "-ar", str(sr),
                        str(wav_out)], capture_output=True)
        out, _ = sf.read(str(wav_out), always_2d=False)
        if out.ndim == 2:
            out = out.mean(axis=1)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def attack_telephone(wm, sr):
    with tempfile.TemporaryDirectory() as td:
        wav_in, mulaw, wav_out = Path(td) / "in.wav", Path(td) / "c.wav", Path(td) / "out.wav"
        sf.write(str(wav_in), wm, sr)
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_in), "-af",
                        "highpass=f=300,lowpass=f=3400", "-ar", "8000",
                        "-c:a", "pcm_mulaw", str(mulaw)], capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(mulaw), "-ar", str(sr),
                        str(wav_out)], capture_output=True)
        out, _ = sf.read(str(wav_out), always_2d=False)
        if out.ndim == 2:
            out = out.mean(axis=1)
        return np.clip(out.astype(np.float32), -1.0, 1.0)


def attack_resample_roundtrip(wm, sr, mid_sr):
    up = librosa.resample(wm, orig_sr=sr, target_sr=mid_sr)
    down = librosa.resample(up, orig_sr=mid_sr, target_sr=sr)
    return np.clip(down.astype(np.float32), -1.0, 1.0)


def attack_crop_with_mask(wm, mask, orig, ratio):
    """Crop은 원본에서도 같은 구간을 잘라내면 시간축이 샘플 단위로 다시 맞으므로,
    잘라낸 원본을 품질 지표(SI-SNR/ViSQOL/PESQ)의 기준 신호로 함께 반환한다."""
    n = len(wm)
    cut_len = int(n * ratio)
    start = np.random.randint(0, max(1, n - cut_len))
    audio_out = np.concatenate([wm[:start], wm[start + cut_len:]]).astype(np.float32)
    mask_out = np.concatenate([mask[:start], mask[start + cut_len:]])
    n_orig = len(orig)
    start_o = min(start, max(0, n_orig - cut_len))
    orig_aligned = np.concatenate([orig[:start_o], orig[start_o + cut_len:]]).astype(np.float32)
    return audio_out, mask_out, orig_aligned, audio_out, True   # 샘플 단위 정합 -> SI-SNR도 신뢰 가능


def attack_tsm_with_mask(wm, mask, orig, rate):
    """TSM은 알려진 배속을 역재생하면 원래 길이로는 복원되지만, phase vocoder 특성상
    구간별로 남는 위상 밀림(local time drift)이 일정하지 않아 샘플 단위 정렬이 보장되지
    않는다. 그래서 de-warp한 신호로 ViSQOL/PESQ(지각 기반, 정렬에 어느 정도 관대함)는
    계산하되, 정렬에 극도로 민감한 SI-SNR은 신뢰할 수 없다고 표시한다."""
    audio_out = librosa.effects.time_stretch(wm, rate=rate).astype(np.float32)
    n_out = len(audio_out)
    idx = np.round(np.linspace(0, len(mask) - 1, n_out)).astype(int)
    mask_out = mask[idx]
    dewarped = librosa.effects.time_stretch(audio_out, rate=1.0 / rate).astype(np.float32)
    return audio_out, mask_out, orig, dewarped, False   # SI-SNR 신뢰 불가 (국소 위상 밀림)


def attack_awgn(wm, snr_db):
    signal_power = np.mean(wm ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(wm))
    return np.clip(wm + noise, -1.0, 1.0).astype(np.float32)


def attack_limiter(wm, sr, threshold_db, attack_ms=5, release_ms=50):
    """룩어헤드 없는 단순 feed-forward 피크 리미터 (envelope follower 기반)."""
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
    limited = wm * gain
    return np.clip(limited, -1.0, 1.0).astype(np.float32)


def build_attacks(sr):
    attacks = []
    for br in MP3_BITRATES:
        attacks.append({"name": f"MP3_{br}kbps", "desync": False,
                         "fn": lambda wm, br=br: attack_mp3(wm, sr, br)})
    attacks.append({"name": f"Resample_{RESAMPLE_ROUNDTRIP_SR}RT", "desync": False,
                     "fn": lambda wm: attack_resample_roundtrip(wm, sr, RESAMPLE_ROUNDTRIP_SR)})
    attacks.append({"name": f"Crop_{int(CROP_RATIO*100)}pct", "desync": True, "combined": True,
                     "fn": lambda wm, mask, orig: attack_crop_with_mask(wm, mask, orig, CROP_RATIO)})
    attacks.append({"name": f"TSM_{TSM_RATE}x", "desync": True, "combined": True,
                     "fn": lambda wm, mask, orig: attack_tsm_with_mask(wm, mask, orig, TSM_RATE)})
    attacks.append({"name": f"AWGN_{AWGN_SNR_DB}dB", "desync": False,
                     "fn": lambda wm: attack_awgn(wm, AWGN_SNR_DB)})
    attacks.append({"name": "Limiter", "desync": False,
                     "fn": lambda wm: attack_limiter(wm, sr, LIMITER_THRESHOLD_DB)})
    attacks.append({"name": f"Opus_{OPUS_BITRATE_KBPS}kbps", "desync": False,
                     "fn": lambda wm: attack_opus(wm, sr, OPUS_BITRATE_KBPS)})
    attacks.append({"name": f"AAC_{AAC_BITRATE_KBPS}kbps", "desync": False,
                    "fn": lambda wm: attack_aac(wm, sr, AAC_BITRATE_KBPS)})
    attacks.append({"name": "TelephoneFilter", "desync": False,
                    "fn": lambda wm: attack_telephone(wm, sr)})
    attacks.append({"name": "EnCodec_6kbps", "desync": False,
                     "fn": lambda wm: attack_encodec(wm, sr)})
    attacks.append({"name": "DAC", "desync": False,
                    "fn": lambda wm: attack_dac(wm, sr)})
    return attacks


# ============================================================
#  1개 파일에 대한 지표 1행 생성
# ============================================================
def build_row(rel_name, stage, orig16k, mask_gt, attacked_audio, detector, sr,
              desync, content_features, spec_out=None, quality_ref=None, quality_deg=None,
              sisnr_reliable=True):
    """quality_ref/quality_deg: Crop/TSM처럼 시간축이 어긋나는 공격에서, 탐지에는 그대로의
    attacked_audio를 쓰되 SI-SNR/ViSQOL/PESQ 계산에는 시간정렬된 신호 쌍을 별도로 넘긴다.
    sisnr_reliable=False면(TSM처럼 국소 위상 밀림이 남는 경우) ViSQOL/PESQ는 계산하되
    정렬에 민감한 SI-SNR만 비워 둔다."""
    ber, nc, dsr, detect_prob, pred_mask = decode_and_measure(detector, attacked_audio, sr)
    loc = localization_metrics(mask_gt, pred_mask)

    ref_q = quality_ref if quality_ref is not None else orig16k
    deg_q = quality_deg if quality_deg is not None else attacked_audio
    can_measure_quality = (not desync) or (quality_ref is not None and quality_deg is not None)

    sisnr = visqol_score = pesq_score = odg_score = None
    if can_measure_quality:
        min_len = min(len(ref_q), len(deg_q))
        if sisnr_reliable:
            sisnr = si_snr(ref_q[:min_len], deg_q[:min_len])
        visqol_score = compute_visqol(ref_q[:min_len], deg_q[:min_len], sr)
        pesq_score = compute_pesq(ref_q[:min_len], deg_q[:min_len], sr)
        odg_score = compute_odg(ref_q[:min_len], deg_q[:min_len], sr)

    if spec_out is not None:
        min_len = min(len(ref_q), len(deg_q))
        save_diff_spectrogram(ref_q[:min_len], deg_q[:min_len], sr, spec_out)

    return {
        "file": rel_name,
        "stage": stage,
        "BER": round(ber, 4),
        "NC": round(nc, 4),
        "DSR": dsr,
        "detect_prob": round(detect_prob, 4),
        "SI-SNR(dB)": round(sisnr, 2) if sisnr is not None else "",
        "ViSQOL": round(visqol_score, 3) if visqol_score is not None else "",
        "PESQ": round(pesq_score, 3) if pesq_score is not None else "",
        "ODG": round(odg_score, 3) if odg_score is not None else "",
        "IoU": round(loc["IoU"], 4),
        "MIoU": round(loc["MIoU"], 4),
        "F1": round(loc["F1"], 4),
        "SpectralFlatness": round(content_features["SpectralFlatness"], 6),
        "SpectralCentroid_Hz": round(content_features["SpectralCentroid_Hz"], 2),
        "EffectiveBandwidth99_Hz": round(content_features["EffectiveBandwidth99_Hz"], 2),
        "SilenceRatio": round(content_features["SilenceRatio"], 6),
        "CrestFactor": round(content_features["CrestFactor"], 4),
    }


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
        print("[ERROR] ffmpeg가 설치되어 있지 않습니다! MP3/Opus 공격을 실행할 수 없습니다.")
        return

    print(f"AudioSeal 탐지기 로딩 중... (device={DEVICE})")
    detector = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE)
    detector.eval()
    generator = None   # 새로 임베딩할 파일이 있을 때만 지연 로딩
    print("완료!\n")

    wav_files = sorted(INPUT_DIR.rglob("*.wav"))
    print(f"총 원본 파일 수: {len(wav_files)}개")
    print("=" * 60)

    attacks = build_attacks(SAMPLE_RATE)
    print(f"공격 단계: {[a['name'] for a in attacks]}")
    spec_saved = 0
    all_rows = []
    content_rows = []
    fieldnames = ["file", "stage", "BER", "NC", "DSR", "detect_prob", "SI-SNR(dB)", "ViSQOL", "PESQ", "ODG",
                  "IoU", "MIoU", "F1", "SpectralFlatness", "SpectralCentroid_Hz",
                  "EffectiveBandwidth99_Hz", "SilenceRatio", "CrestFactor"]
    content_fieldnames = ["file", "SpectralFlatness", "SpectralCentroid_Hz",
                          "EffectiveBandwidth99_Hz", "SilenceRatio", "CrestFactor"]
    csv_path = RESULT_DIR / "pipeline_full_results.csv"

    for i, p in enumerate(wav_files):
        wait_for_safe_temp()
        rel = p.relative_to(INPUT_DIR)
        print(f"\n[{i+1}/{len(wav_files)}] {rel}")
        try:
            # ---------- STEP 1: 전처리 + 임베딩 (이미 임베딩된 파일은 재사용) ----------
            orig16k_path = ORIG16K_DIR / rel.parent / rel.name
            wm_path = EMBED_DIR / rel.parent / f"wm_{rel.name}"

            if orig16k_path.exists() and wm_path.exists():
                orig16k, _ = sf.read(str(orig16k_path), always_2d=False)
                wm16, _ = sf.read(str(wm_path), always_2d=False)
                orig16k = orig16k.astype(np.float32)
                wm16 = wm16.astype(np.float32)
                print(f"  [CACHE]  캐시된 임베딩 재사용 ({len(orig16k)/SAMPLE_RATE:.1f}초)")
            else:
                if generator is None:
                    print("  AudioSeal 생성기 로딩 중 (새로 임베딩할 파일 발견)...")
                    generator = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)
                    generator.eval()

                raw, raw_sr = load_audio_raw(p)
                orig16k = to_model_rate(raw, raw_sr)

                audio_tensor = torch.tensor(orig16k).unsqueeze(0).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    wm_tensor = generator(audio_tensor, message=FIXED_MESSAGE.to(DEVICE), sample_rate=SAMPLE_RATE)
                wm16 = np.clip(wm_tensor.squeeze().cpu().numpy().astype(np.float32), -1.0, 1.0)

                orig16k_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(orig16k_path), orig16k, SAMPLE_RATE, subtype="PCM_16")
                wm_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(wm_path), wm16, SAMPLE_RATE, subtype="PCM_16")
                print(f"  [OK] 임베딩 완료 (원본 sr={raw_sr} -> 16kHz, {len(orig16k)/SAMPLE_RATE:.1f}초)")

            content_features = extract_content_features(orig16k, SAMPLE_RATE)
            content_rows.append({"file": str(rel), **content_features})
            gt_mask = build_gt_mask(len(wm16))

            # ---------- STEP 2: 무공격 지표 ----------
            spec_out = None
            if SAVE_DIFF_SPECTROGRAM and (MAX_SPECTROGRAM_SAVE is None or spec_saved < MAX_SPECTROGRAM_SAVE):
                spec_out = SPEC_DIR / f"{rel.stem}_NoAttack_diff.png"
                spec_saved += 1
            row = build_row(str(rel), "NoAttack", orig16k, gt_mask, wm16, detector, SAMPLE_RATE,
                             desync=False, content_features=content_features, spec_out=spec_out)
            all_rows.append(row)
            print(f"  [NoAttack]  BER={row['BER']} NC={row['NC']} DSR={row['DSR']} "
                  f"MIoU={row['MIoU']} SI-SNR={row['SI-SNR(dB)']}dB")

            # ---------- STEP 3: 공격별 지표 ----------
            for atk in attacks:
                wait_for_safe_temp()
                try:
                    quality_ref = quality_deg = None
                    sisnr_reliable = True
                    if atk.get("combined"):
                        attacked, mask_for_stage, quality_ref, quality_deg, sisnr_reliable = \
                            atk["fn"](wm16, gt_mask, orig16k)
                    else:
                        attacked = atk["fn"](wm16)
                        mask_for_stage = gt_mask
                    if attacked is None:   # 예: EnCodec 실행 실패
                        continue

                    spec_out = None
                    if SAVE_DIFF_SPECTROGRAM and \
                       (MAX_SPECTROGRAM_SAVE is None or spec_saved < MAX_SPECTROGRAM_SAVE):
                        spec_out = SPEC_DIR / f"{rel.stem}_{atk['name']}_diff.png"
                        spec_saved += 1

                    row = build_row(str(rel), atk["name"], orig16k, mask_for_stage, attacked, detector,
                                     SAMPLE_RATE, desync=atk["desync"],
                                     content_features=content_features, spec_out=spec_out,
                                     quality_ref=quality_ref, quality_deg=quality_deg,
                                     sisnr_reliable=sisnr_reliable)
                    all_rows.append(row)
                    print(f"  [{atk['name']}] BER={row['BER']} NC={row['NC']} DSR={row['DSR']} "
                          f"MIoU={row['MIoU']} SI-SNR={row['SI-SNR(dB)']}dB")
                except Exception as e:
                    print(f"  [FAIL] [{atk['name']}] 공격 실패: {e}")

        except Exception as e:
            print(f"  [FAIL] 오류: {rel} → {e}")

        # 파일 하나 끝날 때마다 CSV를 다시 써서, 중간에 죽어도 그때까지 결과는 남도록 함
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        with open(CONTENT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=content_fieldnames)
            writer.writeheader()
            writer.writerows(content_rows)

    # ---------- 결과 CSV (최종) ----------
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with open(CONTENT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=content_fieldnames)
        writer.writeheader()
        writer.writerows(content_rows)

    # ---------- 단계별 요약 ----------
    stages = list(dict.fromkeys(r["stage"] for r in all_rows))   # 등장 순서 유지
    summary_rows = []
    for stage in stages:
        stage_rows = [r for r in all_rows if r["stage"] == stage]
        ber_vals = [r["BER"] for r in stage_rows]
        nc_vals = [r["NC"] for r in stage_rows]
        dsr_vals = [r["DSR"] for r in stage_rows]
        sisnr_vals = [r["SI-SNR(dB)"] for r in stage_rows if r["SI-SNR(dB)"] != ""]
        visqol_vals = [r["ViSQOL"] for r in stage_rows if r["ViSQOL"] != ""]
        pesq_vals = [r["PESQ"] for r in stage_rows if r["PESQ"] != ""]
        miou_vals = [r["MIoU"] for r in stage_rows if r["MIoU"] != ""]
        f1_vals = [r["F1"] for r in stage_rows if r["F1"] != ""]

        dsr_rate = np.mean(dsr_vals) if dsr_vals else 0.0
        summary_rows.append({
            "stage": stage,
            "파일수": len(stage_rows),
            "평균BER": round(float(np.mean(ber_vals)), 4) if ber_vals else "",
            "평균NC": round(float(np.mean(nc_vals)), 4) if nc_vals else "",
            "DSR(%)": round(dsr_rate * 100, 1),
            "평균SI-SNR(dB)": round(float(np.mean(sisnr_vals)), 2) if sisnr_vals else "",
            "평균ViSQOL": round(float(np.mean(visqol_vals)), 3) if visqol_vals else "",
            "평균PESQ": round(float(np.mean(pesq_vals)), 3) if pesq_vals else "",
            "평균MIoU": round(float(np.mean(miou_vals)), 4) if miou_vals else "",
            "평균F1": round(float(np.mean(f1_vals)), 4) if f1_vals else "",
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
    print(f"  {'단계':<20} {'BER':<8} {'DSR':<8} {'MIoU':<8} {'SI-SNR':<10} 판정")
    print(f"  {'-'*65}")
    for s in summary_rows:
        print(f"  {s['stage']:<20} {s['평균BER']:<8} {s['DSR(%)']}%{'':<4} "
              f"{s['평균MIoU']:<8} {s['평균SI-SNR(dB)']}dB{'':<4} {s['판정']}")
    print("=" * 60)
    print(f"  전체 결과 CSV: {csv_path}")
    print(f"  콘텐츠 특성 CSV: {CONTENT_CSV}")
    print(f"  요약 CSV     : {summary_path}")
    print(f"  임베딩 결과  : {EMBED_DIR}")
    print(f"  16k 원본     : {ORIG16K_DIR}")
    if SAVE_DIFF_SPECTROGRAM:
        print(f"  차이 스펙트로그램({spec_saved}개): {SPEC_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
