r"""
실행 전 환경 확인
------------------
# tmux 세션 방식 (추천 — 나중에 다시 접속해서 진행 상황 볼 수 있음)
tmux new -s wm
python3 /home/hansung1/.vscode-server/audio/sc44_pipeline_full.py
# Ctrl+B, D 로 detach 하고 노트북 닫아도 됨
# 나중에: tmux attach -t wm

# 또는 nohup 방식
nohup python3 /home/hansung1/.vscode-server/audio/sc44_pipeline_full.py > pipeline.log 2>&1 &
disown

VS Code에서 `Ctrl + Shift + P` → `Python: Select Interpreter`를 실행한 뒤
기존에 쓰던 SilentCipher 가상환경(`/home/hansung1/.vscode-server/audio/venv`)을 선택하세요.
MP3/Opus 공격에는 시스템 `ffmpeg`도 필요합니다.

참고: SC-16과 SC-44는 `audioseal`을 사용하지 않습니다.
`visqol`, `encodec`, `descript-audio-codec`는 전체 평가에 필요한 패키지입니다.
따라서 AudioSeal용 `audioseal` 진단은 SC-16/SC-44에서는 발생하지 않습니다.

SilentCipher SC-44 통합 파이프라인 — 임베딩 -> 무공격 지표 -> 공격 지표 (원스톱)
====================================================================
as_pipeline_full.py(AudioSeal), wavmark_pipeline_full.py(WavMark)와 동일한 구조.
비교 기준은 항상 "원본"(mono+44.1kHz 전처리본)이며, SilentCipher 전용 심화지표로
위상 일관성(Phase Locking Value)·그룹 지연(Group Delay) 오차를 추가했다.

※ 44.1kHz 음악 데이터는 SC-44(44.1kHz 모델)로만 처리한다 — 16kHz로 내리지 않는다.
   (SC-16이 필요하면 sc16_pipeline_full.py를 별도로 둘 것)

STEP 1. original/ 의 원본을 mono+44.1kHz로 변환해 SilentCipher SC-44 워터마크 삽입
STEP 2. 무공격 지표: BER, NC, DSR, 위상일관성(PLV)·그룹지연(GD MAE), SI-SNR, ViSQOL, ODG, 차이 스펙트로그램
    + 콘텐츠 특성 5개: Spectral Flatness, Spectral Centroid, Effective Bandwidth(99%), Silence Ratio, Crest Factor
STEP 3. 아래 5개 공격 카테고리(총 8개 조건)를 적용한 뒤 동일 지표 재계산
    - 압축        : MP3 128kbps / 320kbps
    - 추가 압축/전화망: AAC 128kbps, TelephoneFilter
    - 형식 변환   : 리샘플링 44.1kHz -> 16kHz -> 44.1kHz
    - 동기화 파괴 : Cropping(10% 무작위 절삭), TSM(속도 10% 변조)
    - 잡음/왜곡   : AWGN(SNR 30dB), Limiter
    - 차세대 코덱 : Opus(64kbps) [EnCodec/DAC은 설치돼 있으면 자동 추가]

위상 일관성·그룹 지연은 SilentCipher가 "위상은 그대로 두고 진폭에만 심는다"고
설계된 기법이라는 주장을 검증하는 전용 심화지표다. 원본과 워터마크(또는 공격 후)
신호의 STFT 위상 스펙트럼을 비교해서:
    PLV(Phase Locking Value)  : 진폭 가중 위상 일관성, 1.0에 가까울수록 위상 보존
    PhaseDiff_rad             : 평균 절대 위상차(라디안), 0에 가까울수록 위상 보존
    GroupDelayMAE             : 주파수축 위상 미분(그룹 지연)의 평균절대오차, 0에 가까울수록 보존
공격 전(NoAttack)에도 계산해서 삽입 자체가 설계 의도대로 위상을 건드리지 않는지
하한선을 먼저 확인한다.

동기화 파괴 공격(Crop, TSM) 이후에는 시간축이 어긋나 샘플/STFT 단위 정렬 비교가
무의미하므로 SI-SNR/ViSQOL/ODG/위상지표는 계산하지 않고 BER/NC/DSR만 기록한다.

* ViSQOL/EnCodec/DAC은 전체 평가에 포함되므로 해당 패키지를 설치해야 한다.

결과: sc44_pipeline_results/pipeline_full_results.csv   (파일 x 단계 전체 결과)
    sc44_pipeline_results/pipeline_full_summary.csv   (단계별 평균 요약)
    sc44_pipeline_results/content_features.csv         (원본 파일별 콘텐츠 특성)
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GPU 1번만 사용

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

import silentcipher

matplotlib.rcParams["font.family"] = "Baekmuk Gulim"   # 한글 깨짐 방지 (Linux)
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
#  설정
# ============================================================
AUDIO_ROOT  = Path("/home/hansung1/.vscode-server/audio")
INPUT_DIR   = AUDIO_ROOT / "original"
EMBED_DIR   = AUDIO_ROOT / "sc44_embed"
RESULT_DIR  = AUDIO_ROOT / "sc44_pipeline_results"
ORIG44K_DIR = RESULT_DIR / "original_44k1_mono"
SPEC_DIR    = RESULT_DIR / "diff_spectrograms"
CONTENT_CSV = RESULT_DIR / "content_features.csv"

SAMPLE_RATE  = 44100          # SC-44 고정 (44.1kHz 음악 데이터는 반드시 이 버전 사용)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE   = "44.1k"
WM_MESSAGE   = [123, 234, 111, 222, 11]   # 5바이트 x 8비트 = 40비트
MESSAGE_BITS = 40

# GPU 온도 제한 (nvidia-smi 기반, 물리적 GPU 인덱스 -- CUDA_VISIBLE_DEVICES와 무관하게
# nvidia-smi --id는 실제 GPU 번호를 그대로 받는다)
GPU_INDEX      = 1
TEMP_LIMIT_C   = 78.0   # 이 온도 이상이면 일시정지
TEMP_RESUME_C  = 70.0   # 이 온도 이하로 내려가야 재개
TEMP_POLL_SEC  = 10

DSR_PASS_RATE = 0.95

SAVE_DIFF_SPECTROGRAM = True   # NoAttack + 모든 공격 단계에서 저장
MAX_SPECTROGRAM_SAVE  = None   # 저장 개수 제한. None이면 무제한

# 공격 파라미터
MP3_BITRATES          = [128, 320]
RESAMPLE_MID_SR        = 16000        # 44.1k -> 16k -> 44.1k 왕복
CROP_RATIO             = 0.10
TSM_RATE               = 1.10
AWGN_SNR_DB            = 30
LIMITER_THRESHOLD_DB   = -6.0
OPUS_BITRATE_KBPS      = 64           # 44.1kHz 풀밴드 음악이라 32kbps보다 여유있게 설정
AAC_BITRATE_KBPS       = 128

FEATURE_N_FFT          = 1024
FEATURE_HOP_LENGTH     = 256
SILENCE_THRESHOLD_DB   = -40.0

np.random.seed(42)

# ============================================================
#  ViSQOL / ODG (선택적 — 설치돼 있을 때만 계산, 없으면 자동 스킵)
#  ViSQOL의 audio(음악) 모드는 48kHz 기준이라 내부적으로 48kHz로 리샘플링해서 호출한다.
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
#  ODG(PEAQ)의 실질적인 대체 지표. wideband 모드는 16kHz 입력을 요구하므로
#  44.1kHz 신호는 항상 16kHz로 리샘플링해서 계산한다.
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
    """mono + SC-44가 요구하는 44.1kHz로 변환."""
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


def message_to_bits(msg):
    bits = []
    for byte_val in msg:
        for b in range(8):
            bits.append((byte_val >> b) & 1)
    return np.array(bits, dtype=np.uint8)


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
    n_fft, hop = 2048, 512
    S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig, n_fft=n_fft, hop_length=hop)), ref=np.max)
    S_wm = librosa.amplitude_to_db(np.abs(librosa.stft(wm, n_fft=n_fft, hop_length=hop)), ref=np.max)
    S_diff = S_wm - S_orig

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, S, title, cmap in zip(
        axes, [S_orig, S_wm, S_diff],
        ["원본(44.1k)", "워터마크 삽입/공격", "차이 (대상 - 원본)"],
        ["magma", "magma", "coolwarm"],
    ):
        img = librosa.display.specshow(S, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=ax, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def phase_consistency_metrics(orig, deg, sr, n_fft=2048, hop=512):
    """SilentCipher 전용 심화지표: 위상 일관성(PLV) + 그룹 지연 오차."""
    min_len = min(len(orig), len(deg))
    orig = orig[:min_len]
    deg = deg[:min_len]

    S_o = librosa.stft(orig, n_fft=n_fft, hop_length=hop)
    S_d = librosa.stft(deg, n_fft=n_fft, hop_length=hop)
    n_frames = min(S_o.shape[1], S_d.shape[1])
    S_o, S_d = S_o[:, :n_frames], S_d[:, :n_frames]

    phase_o = np.angle(S_o)
    phase_d = np.angle(S_d)
    mag_w = np.abs(S_o)   # 원본 진폭으로 가중 (청감상 중요한 성분 위주로 평가)

    phase_diff = phase_d - phase_o
    plv = np.abs(np.sum(mag_w * np.exp(1j * phase_diff))) / (np.sum(mag_w) + 1e-9)
    mean_abs_phase_diff = float(np.mean(np.abs(np.angle(np.exp(1j * phase_diff)))))

    gd_o = -np.diff(np.unwrap(phase_o, axis=0), axis=0)
    gd_d = -np.diff(np.unwrap(phase_d, axis=0), axis=0)
    group_delay_mae = float(np.mean(np.abs(gd_d - gd_o)))

    return {"PLV": float(plv), "PhaseDiff_rad": mean_abs_phase_diff, "GroupDelayMAE": group_delay_mae}


# ============================================================
#  SilentCipher 디코딩 / BER / NC / DSR
# ============================================================
def decode_and_measure(model, audio, sr):
    result = model.decode_wav(audio, sr, phase_shift_decoding=False)
    detected = bool(result["status"])
    msg = result["messages"][0] if result["messages"] else None
    conf = result["confidences"][0] if result["confidences"] else 0.0

    orig_bits = message_to_bits(WM_MESSAGE)
    if msg is not None:
        dec_bits = message_to_bits(msg)
        ber = float(np.mean(dec_bits != orig_bits))
        nc = normalized_correlation(orig_bits, dec_bits)
    else:
        ber = 1.0
        nc = -1.0

    dsr = 1 if detected else 0
    return ber, nc, dsr, float(conf)


# ============================================================
#  공격 함수 (44.1kHz mono 입력 -> 44.1kHz mono 출력)
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
    down = librosa.resample(wm, orig_sr=sr, target_sr=mid_sr)
    up = librosa.resample(down, orig_sr=mid_sr, target_sr=sr)
    return np.clip(up.astype(np.float32), -1.0, 1.0)


def attack_crop(wm, orig, ratio):
    """원본에서도 같은 구간을 잘라내면 시간축이 샘플 단위로 다시 맞으므로,
    잘라낸 원본을 품질 지표(SI-SNR/ViSQOL/PESQ)의 기준 신호로 함께 반환한다."""
    n = len(wm)
    cut_len = int(n * ratio)
    start = np.random.randint(0, max(1, n - cut_len))
    audio_out = np.concatenate([wm[:start], wm[start + cut_len:]]).astype(np.float32)
    n_orig = len(orig)
    start_o = min(start, max(0, n_orig - cut_len))
    orig_aligned = np.concatenate([orig[:start_o], orig[start_o + cut_len:]]).astype(np.float32)
    return audio_out, orig_aligned, audio_out, True   # 샘플 단위 정합 -> SI-SNR도 신뢰 가능


def attack_tsm(wm, orig, rate):
    """알려진 배속을 역재생하면 길이는 복원되지만 phase vocoder 특성상 구간별 위상
    밀림이 남아 샘플 단위 정렬은 보장되지 않는다. de-warp한 신호로 ViSQOL/PESQ는
    계산하되, 정렬에 극도로 민감한 SI-SNR은 신뢰할 수 없다고 표시한다."""
    audio_out = librosa.effects.time_stretch(wm, rate=rate).astype(np.float32)
    dewarped = librosa.effects.time_stretch(audio_out, rate=1.0 / rate).astype(np.float32)
    return audio_out, orig, dewarped, False   # SI-SNR 신뢰 불가 (국소 위상 밀림)


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
    return np.clip(wm * gain, -1.0, 1.0).astype(np.float32)


def build_attacks(sr):
    attacks = []
    for br in MP3_BITRATES:
        attacks.append({"name": f"MP3_{br}kbps", "desync": False,
                         "fn": lambda wm, br=br: attack_mp3(wm, sr, br)})
    attacks.append({"name": f"Resample_{sr}-{RESAMPLE_MID_SR}RT", "desync": False,
                     "fn": lambda wm: attack_resample_roundtrip(wm, sr, RESAMPLE_MID_SR)})
    attacks.append({"name": f"Crop_{int(CROP_RATIO*100)}pct", "desync": True, "combined": True,
                     "fn": lambda wm, orig: attack_crop(wm, orig, CROP_RATIO)})
    attacks.append({"name": f"TSM_{TSM_RATE}x", "desync": True, "combined": True,
                     "fn": lambda wm, orig: attack_tsm(wm, orig, TSM_RATE)})
    attacks.append({"name": f"AWGN_{AWGN_SNR_DB}dB", "desync": False,
                     "fn": lambda wm: attack_awgn(wm, AWGN_SNR_DB)})
    attacks.append({"name": "Limiter", "desync": False,
                     "fn": lambda wm: attack_limiter(wm, sr, LIMITER_THRESHOLD_DB)})
    attacks.append({"name": f"Opus_{OPUS_BITRATE_KBPS}kbps", "desync": False,
                     "fn": lambda wm: attack_opus(wm, sr, OPUS_BITRATE_KBPS)})
    # AudioSeal/SC-16 실험과 동일한 비트레이트(32kbps)로도 추가 측정 -> 64kbps만 있던
    # 비대칭 문제(SC-44 직접 비교 공정성 훼손) 보완
    attacks.append({"name": "Opus_32kbps", "desync": False,
                     "fn": lambda wm: attack_opus(wm, sr, 32)})
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
def build_row(rel_name, stage, orig44k, attacked_audio, model, sr, desync,
              content_features, spec_out=None, quality_ref=None, quality_deg=None,
              sisnr_reliable=True):
    """quality_ref/quality_deg: Crop/TSM처럼 시간축이 어긋나는 공격에서, 탐지에는 그대로의
    attacked_audio를 쓰되 SI-SNR/ViSQOL/PESQ/위상 지표 계산에는 시간정렬된 신호 쌍을
    별도로 넘긴다. sisnr_reliable=False면(TSM처럼 국소 위상 밀림이 남는 경우) ViSQOL/PESQ는
    계산하되 정렬에 극도로 민감한 SI-SNR·위상 지표는 비워 둔다."""
    ber, nc, dsr, confidence = decode_and_measure(model, attacked_audio, sr)

    ref_q = quality_ref if quality_ref is not None else orig44k
    deg_q = quality_deg if quality_deg is not None else attacked_audio
    can_measure_quality = (not desync) or (quality_ref is not None and quality_deg is not None)

    sisnr = visqol_score = pesq_score = odg_score = None
    plv = phase_diff = gd_mae = None
    if can_measure_quality:
        min_len = min(len(ref_q), len(deg_q))
        visqol_score = compute_visqol(ref_q[:min_len], deg_q[:min_len], sr)
        pesq_score = compute_pesq(ref_q[:min_len], deg_q[:min_len], sr)
        odg_score = compute_odg(ref_q[:min_len], deg_q[:min_len], sr)
        if sisnr_reliable:
            sisnr = si_snr(ref_q[:min_len], deg_q[:min_len])
            ph = phase_consistency_metrics(ref_q[:min_len], deg_q[:min_len], sr)
            plv, phase_diff, gd_mae = ph["PLV"], ph["PhaseDiff_rad"], ph["GroupDelayMAE"]

    if spec_out is not None:
        min_len = min(len(ref_q), len(deg_q))
        save_diff_spectrogram(ref_q[:min_len], deg_q[:min_len], sr, spec_out)

    return {
        "file": rel_name,
        "stage": stage,
        "BER": round(ber, 4),
        "NC": round(nc, 4),
        "DSR": dsr,
        "confidence": round(confidence, 4),
        "SI-SNR(dB)": round(sisnr, 2) if sisnr is not None else "",
        "ViSQOL": round(visqol_score, 3) if visqol_score is not None else "",
        "PESQ": round(pesq_score, 3) if pesq_score is not None else "",
        "ODG": round(odg_score, 3) if odg_score is not None else "",
        "PLV": round(plv, 4) if plv is not None else "",
        "PhaseDiff_rad": round(phase_diff, 4) if phase_diff is not None else "",
        "GroupDelayMAE": round(gd_mae, 4) if gd_mae is not None else "",
        "SpectralFlatness": round(content_features["SpectralFlatness"], 6),
        "SpectralCentroid_Hz": round(content_features["SpectralCentroid_Hz"], 2),
        "EffectiveBandwidth99_Hz": round(content_features["EffectiveBandwidth99_Hz"], 2),
        "SilenceRatio": round(content_features["SilenceRatio"], 6),
        "CrestFactor": round(content_features["CrestFactor"], 4),
    }


# ============================================================
#  재개(resume) 지원
# ============================================================
def _cast_numeric(value):
    """CSV에서 다시 읽은 문자열 값을 숫자로 복원 (빈 문자열은 그대로 둠)."""
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
                if key not in ("file", "stage"):
                    row[key] = _cast_numeric(row[key])
            rows.append(row)
    return rows


# ============================================================
#  메인
# ============================================================
def main():
    for d in (EMBED_DIR, RESULT_DIR, ORIG44K_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if SAVE_DIFF_SPECTROGRAM:
        SPEC_DIR.mkdir(parents=True, exist_ok=True)

    ffmpeg_ok = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    if not ffmpeg_ok:
        print("[ERROR] ffmpeg가 설치되어 있지 않습니다! MP3/Opus 공격을 실행할 수 없습니다.")
        return

    print("SilentCipher SC-44 모델 로딩 중...")
    model = silentcipher.get_model(model_type=MODEL_TYPE, device=DEVICE)
    print("완료!\n")
    print(f"메시지: {WM_MESSAGE} ({MESSAGE_BITS}비트)")

    wav_files = sorted(INPUT_DIR.rglob("*.wav"))
    print(f"총 원본 파일 수: {len(wav_files)}개")
    print("=" * 60)

    attacks = build_attacks(SAMPLE_RATE)
    expected_stages = {"NoAttack"} | {a["name"] for a in attacks}
    print(f"공격 단계: {[a['name'] for a in attacks]}")
    print("※ 44.1kHz 전체 트랙을 그대로 처리하므로 Limiter처럼 샘플 단위 루프가 있는")
    print("   공격은 파일당 다소 시간이 걸릴 수 있습니다.")
    print("=" * 60)

    fieldnames = ["file", "stage", "BER", "NC", "DSR", "confidence", "SI-SNR(dB)", "ViSQOL", "PESQ", "ODG",
                  "PLV", "PhaseDiff_rad", "GroupDelayMAE", "SpectralFlatness",
                  "SpectralCentroid_Hz", "EffectiveBandwidth99_Hz", "SilenceRatio", "CrestFactor"]
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
            # ---------- STEP 1: 전처리 + 임베딩 (이미 임베딩된 파일은 캐시 재사용) ----------
            orig44k_path = ORIG44K_DIR / rel.parent / rel.name
            wm_path = EMBED_DIR / rel.parent / f"wm_{rel.name}"

            if orig44k_path.exists() and wm_path.exists():
                orig44k, _ = sf.read(str(orig44k_path), always_2d=False)
                wm44k, _ = sf.read(str(wm_path), always_2d=False)
                orig44k = orig44k.astype(np.float32)
                wm44k = wm44k.astype(np.float32)
                print(f"  [CACHE]  캐시된 임베딩 재사용 ({len(orig44k)/SAMPLE_RATE:.1f}초)")
            else:
                raw, raw_sr = load_audio_raw(p)
                orig44k = to_model_rate(raw, raw_sr)

                wm44k, sdr = model.encode_wav(orig44k, SAMPLE_RATE, WM_MESSAGE)
                wm44k = np.clip(np.asarray(wm44k).astype(np.float32).squeeze(), -1.0, 1.0)

                orig44k_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(orig44k_path), orig44k, SAMPLE_RATE, subtype="PCM_16")
                wm_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(wm_path), wm44k, SAMPLE_RATE, subtype="PCM_32")
                print(f"  [OK] 임베딩 완료 (원본 sr={raw_sr} -> 44.1kHz, {len(orig44k)/SAMPLE_RATE:.1f}초, "
                      f"encode SDR={sdr:.2f}dB)")

            content_features = extract_content_features(orig44k, SAMPLE_RATE)

            # ---------- STEP 2: 무공격 지표 ----------
            spec_out = None
            if SAVE_DIFF_SPECTROGRAM and (MAX_SPECTROGRAM_SAVE is None or spec_saved < MAX_SPECTROGRAM_SAVE):
                spec_out = SPEC_DIR / f"{rel.stem}_NoAttack_diff.png"
                spec_saved += 1
            row = build_row(str(rel), "NoAttack", orig44k, wm44k, model, SAMPLE_RATE,
                             desync=False, content_features=content_features, spec_out=spec_out)
            all_rows.append(row)
            print(f"  [NoAttack]  BER={row['BER']} NC={row['NC']} DSR={row['DSR']} "
                  f"PLV={row['PLV']} SI-SNR={row['SI-SNR(dB)']}dB")

            # ---------- STEP 3: 공격별 지표 ----------
            for atk in attacks:
                wait_for_safe_temp()
                try:
                    quality_ref = quality_deg = None
                    sisnr_reliable = True
                    if atk.get("combined"):
                        attacked, quality_ref, quality_deg, sisnr_reliable = atk["fn"](wm44k, orig44k)
                    else:
                        attacked = atk["fn"](wm44k)
                    if attacked is None:   # 예: EnCodec 실행 실패
                        continue

                    spec_out = None
                    if SAVE_DIFF_SPECTROGRAM and \
                       (MAX_SPECTROGRAM_SAVE is None or spec_saved < MAX_SPECTROGRAM_SAVE):
                        spec_out = SPEC_DIR / f"{rel.stem}_{atk['name']}_diff.png"
                        spec_saved += 1

                    row = build_row(str(rel), atk["name"], orig44k, attacked, model,
                                     SAMPLE_RATE, desync=atk["desync"],
                                     content_features=content_features, spec_out=spec_out,
                                     quality_ref=quality_ref, quality_deg=quality_deg,
                                     sisnr_reliable=sisnr_reliable)
                    all_rows.append(row)
                    print(f"  [{atk['name']}] BER={row['BER']} NC={row['NC']} DSR={row['DSR']} "
                          f"PLV={row['PLV']} SI-SNR={row['SI-SNR(dB)']}dB")
                except Exception as e:
                    print(f"  [FAIL] [{atk['name']}] 공격 실패: {e}")

        except Exception as e:
            print(f"  [FAIL] 오류: {rel} → {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # 44.1kHz 풀트랙 인코딩 후 파편화된 GPU 메모리 회수 (OOM 완화)

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
            "파일수": len(stage_rows),
            "평균BER": round(float(np.mean(numeric("BER"))), 4) if numeric("BER") else "",
            "평균NC": round(float(np.mean(numeric("NC"))), 4) if numeric("NC") else "",
            "DSR(%)": round(dsr_rate * 100, 1),
            "평균SI-SNR(dB)": round(float(np.mean(numeric("SI-SNR(dB)"))), 2) if numeric("SI-SNR(dB)") else "",
            "평균ViSQOL": round(float(np.mean(numeric("ViSQOL"))), 3) if numeric("ViSQOL") else "",
            "평균PESQ": round(float(np.mean(numeric("PESQ"))), 3) if numeric("PESQ") else "",
            "평균PLV": round(float(np.mean(numeric("PLV"))), 4) if numeric("PLV") else "",
            "평균GroupDelayMAE": round(float(np.mean(numeric("GroupDelayMAE"))), 4) if numeric("GroupDelayMAE") else "",
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
    print(f"  {'단계':<24} {'BER':<8} {'DSR':<8} {'PLV':<8} {'SI-SNR':<10} 판정")
    print(f"  {'-'*65}")
    for s in summary_rows:
        print(f"  {s['stage']:<24} {s['평균BER']:<8} {s['DSR(%)']}%{'':<4} "
              f"{s['평균PLV']:<8} {s['평균SI-SNR(dB)']}dB{'':<4} {s['판정']}")
    print("=" * 60)
    print(f"  전체 결과 CSV : {csv_path}")
    print(f"  요약 CSV      : {summary_path}")
    print(f"  콘텐츠 특성 CSV: {CONTENT_CSV}")
    print(f"  임베딩 결과   : {EMBED_DIR}")
    print(f"  44.1k 원본    : {ORIG44K_DIR}")
    if SAVE_DIFF_SPECTROGRAM:
        print(f"  차이 스펙트로그램({spec_saved}개): {SPEC_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
