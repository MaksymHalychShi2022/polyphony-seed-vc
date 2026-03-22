import matplotlib

matplotlib.use("Agg")

import gradio as gr
import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torchaudio
from ten_vad import TenVad

# s3prl (SingMOS dependency) uses APIs removed in torchaudio >= 2.1
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *a, **kw: None
if not hasattr(torchaudio, "sox_effects"):
    import sys
    import types

    _sox = types.ModuleType("torchaudio.sox_effects")
    _sox.apply_effects_tensor = (
        lambda waveform, sample_rate, effects, channels_first=True: (
            waveform,
            sample_rate,
        )
    )
    sys.modules["torchaudio.sox_effects"] = _sox
    torchaudio.sox_effects = _sox


CHUNK_SECONDS = 5
SR = 16000
CHUNK_SAMPLES = CHUNK_SECONDS * SR
VAD_HOP = 256  # samples @ 16kHz ≈ 16 ms per frame

predictor = None


def load_predictor():
    global predictor
    if predictor is None:
        predictor = torch.hub.load(
            "South-Twilight/SingMOS:v1.1.2", "singmos_pro", trust_repo=True
        )
        predictor.eval()
    return predictor


def score_chunks(wave: np.ndarray) -> tuple[list[float], list[float]]:
    model = load_predictor()
    scores, energies = [], []
    for start in range(0, len(wave), CHUNK_SAMPLES):
        chunk = wave[start : start + CHUNK_SAMPLES]
        energies.append(float(np.sqrt(np.mean(chunk**2))))
        wave_t = torch.from_numpy(chunk).unsqueeze(0)
        length = torch.tensor([wave_t.shape[1]], dtype=torch.long)
        with torch.no_grad():
            score = model(wave_t, length)
        scores.append(score.item())
    return scores, energies


def build_mos_plot(scores: list[float], chunk_seconds: int) -> plt.Figure:
    n = len(scores)
    times = [i * chunk_seconds + chunk_seconds / 2 for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 3.5))
    colors = plt.cm.RdYlGn((np.array(scores) - 1.0) / 4.0)
    ax.bar(
        times,
        scores,
        width=chunk_seconds * 0.85,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.plot(
        times,
        scores,
        color="steelblue",
        linewidth=1.5,
        marker="o",
        markersize=4,
        zorder=5,
    )
    ax.set_xlim(0, n * chunk_seconds)
    ax.set_ylim(1.0, 5.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MOS")
    ax.set_title("SingMOS — quality over time")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(chunk_seconds))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def compute_vad(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (times_sec, speech_probabilities) at VAD_HOP resolution."""
    vad = TenVad(hop_size=VAD_HOP)
    wave_int16 = (wave * 32767).astype(np.int16)
    probs = []
    for start in range(0, len(wave_int16) - VAD_HOP + 1, VAD_HOP):
        prob, _ = vad.process(wave_int16[start : start + VAD_HOP])
        probs.append(prob)
    times = np.arange(len(probs)) * VAD_HOP / SR + VAD_HOP / SR / 2
    return times, np.array(probs)


def build_vad_plot(
    vad_times: np.ndarray,
    vad_probs: np.ndarray,
    total_seconds: float,
    chunk_seconds: int,
) -> plt.Figure:
    n_chunks = int(np.ceil(total_seconds / chunk_seconds))
    fig, ax = plt.subplots(figsize=(max(6, n_chunks * 0.8), 3.5))
    ax.fill_between(vad_times, vad_probs, alpha=0.35, color="mediumpurple")
    ax.plot(vad_times, vad_probs, color="mediumpurple", linewidth=1.0)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="threshold 0.5")
    ax.set_xlim(0, total_seconds)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speech probability")
    ax.set_title("Voice Activity Detection (TEN VAD)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(chunk_seconds))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def build_energy_plot(energies: list[float], chunk_seconds: int) -> plt.Figure:
    n = len(energies)
    times = [i * chunk_seconds + chunk_seconds / 2 for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 3.5))
    ax.bar(
        times,
        energies,
        width=chunk_seconds * 0.85,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.7,
    )
    ax.plot(
        times,
        energies,
        color="darkorange",
        linewidth=1.5,
        marker="o",
        markersize=4,
        zorder=5,
    )
    ax.set_xlim(0, n * chunk_seconds)
    ax.set_ylim(0, None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Energy")
    ax.set_title("RMS Energy over time")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(chunk_seconds))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def predict_mos(audio_path: str):
    if audio_path is None:
        return None, None, None, "No audio provided."

    wave, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != SR:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=SR)

    total_seconds = len(wave) / SR
    scores, energies = score_chunks(wave)
    vad_times, vad_probs = compute_vad(wave)
    mean_score = float(np.mean(scores))
    summary = f"Mean MOS: {mean_score:.4f}  |  Chunks: {len(scores)}  |  Duration: {total_seconds:.1f}s"
    return (
        build_mos_plot(scores, CHUNK_SECONDS),
        build_energy_plot(energies, CHUNK_SECONDS),
        build_vad_plot(vad_times, vad_probs, total_seconds, CHUNK_SECONDS),
        summary,
    )


with gr.Blocks(title="SingMOS Scorer") as demo:
    gr.Markdown(
        "# SingMOS Scorer\nPer-chunk MOS quality and energy plots for singing audio."
    )
    audio_input = gr.Audio(type="filepath", label="Input Audio")
    run_btn = gr.Button("Score")
    mos_plot = gr.Plot(label="MOS over time")
    energy_plot = gr.Plot(label="RMS Energy over time")
    vad_plot = gr.Plot(label="Voice Activity Detection")
    summary_output = gr.Textbox(label="Summary", interactive=False)

    run_btn.click(
        predict_mos,
        inputs=audio_input,
        outputs=[mos_plot, energy_plot, vad_plot, summary_output],
    )

if __name__ == "__main__":
    demo.launch()
