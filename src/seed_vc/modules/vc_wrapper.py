from __future__ import annotations

from typing import Callable

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio

DEFAULT_CHECKPOINT_REPO_ID = "MaksymHalych/polyphony-seed-vc"
DEFAULT_CHECKPOINT = "DiT_epoch_00058_step_10000.pth"
DEFAULT_CONFIG_REPO_ID = "Plachta/Seed-VC"
DEFAULT_CONFIG = "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"


class VoiceConversionWrapper(nn.Module):
    def __init__(
        self,
        sr: int,
        hop_size: int,
        mel_fn: Callable[[torch.Tensor], torch.Tensor],
        cfm: nn.Module,
        length_regulator: nn.Module,
        whisper_model: nn.Module,
        whisper_feature_extractor,
        style_encoder: nn.Module,
        f0_extractor,
        vocoder: nn.Module,
    ):
        super().__init__()
        self.sr = sr
        self.hop_size = hop_size
        self.mel_fn = mel_fn
        self.cfm = cfm
        self.length_regulator = length_regulator
        self.whisper_model = whisper_model
        self.whisper_feature_extractor = whisper_feature_extractor
        self.style_encoder = style_encoder
        self.f0_extractor = f0_extractor
        self.vocoder = vocoder
        self.overlap_frame_len = 16
        self.dit_max_context_len = 30  # seconds

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_style(self, waves_16k: torch.Tensor) -> torch.Tensor:
        feat = torchaudio.compliance.kaldi.fbank(
            waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.style_encoder(feat.unsqueeze(0))

    @torch.no_grad()
    def extract_semantic(self, waves_16k: torch.Tensor) -> torch.Tensor:
        if waves_16k.size(-1) <= 16000 * 30:
            return self._whisper_forward(waves_16k)

        overlapping_time = 5
        features_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < waves_16k.size(-1):
            if buffer is None:
                chunk = waves_16k[:, traversed_time : traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [
                        buffer,
                        waves_16k[
                            :,
                            traversed_time : traversed_time
                            + 16000 * (30 - overlapping_time),
                        ],
                    ],
                    dim=-1,
                )
            feat = self._whisper_forward(chunk)
            if traversed_time == 0:
                features_list.append(feat)
            else:
                features_list.append(feat[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed_time += (
                30 * 16000
                if traversed_time == 0
                else chunk.size(-1) - 16000 * overlapping_time
            )
        return torch.cat(features_list, dim=1)

    def _whisper_forward(self, waves_16k: torch.Tensor) -> torch.Tensor:
        inputs = self.whisper_feature_extractor(
            [waves_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = self.whisper_model._mask_input_features(
            inputs.input_features, attention_mask=inputs.attention_mask
        ).to(waves_16k.device)
        outputs = self.whisper_model.encoder(
            input_features.to(self.whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        feat = outputs.last_hidden_state.to(torch.float32)
        return feat[:, : waves_16k.size(-1) // 320 + 1]

    @staticmethod
    def adjust_f0(
        f0_alt: torch.Tensor,
        f0_ori: torch.Tensor,
        auto_adjust: bool,
        pitch_shift: int,
    ) -> torch.Tensor:
        voiced_f0_ori = f0_ori[f0_ori > 1]
        voiced_f0_alt = f0_alt[f0_alt > 1]

        log_f0_alt = torch.log(f0_alt + 1e-5)

        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_adjust and voiced_f0_ori.numel() > 0 and voiced_f0_alt.numel() > 0:
            median_log_f0_ori = torch.log(voiced_f0_ori + 1e-5).median()
            median_log_f0_alt = torch.log(voiced_f0_alt + 1e-5).median()
            shifted_log_f0_alt[f0_alt > 1] = (
                log_f0_alt[f0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            )

        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[f0_alt > 1] = shifted_f0_alt[f0_alt > 1] * (
                2 ** (pitch_shift / 12)
            )
        return shifted_f0_alt

    @staticmethod
    def crossfade(chunk1: np.ndarray, chunk2: np.ndarray, overlap: int) -> np.ndarray:
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    @torch.inference_mode()
    def convert(
        self,
        source_path: str,
        target_path: str,
        diffusion_steps: int = 10,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.7,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> np.ndarray:
        # 3.1 — load and resample audio
        source_wave = librosa.load(source_path, sr=self.sr)[0]
        target_wave = librosa.load(target_path, sr=self.sr)[0]
        target_wave = target_wave[: self.sr * 25]

        source_tensor = torch.tensor(source_wave).unsqueeze(0).float().to(device)
        target_tensor = torch.tensor(target_wave).unsqueeze(0).float().to(device)

        source_16k = torchaudio.functional.resample(source_tensor, self.sr, 16000)
        target_16k = torchaudio.functional.resample(target_tensor, self.sr, 16000)

        # 3.2 — extract features
        S_alt = self.extract_semantic(source_16k)
        S_ori = self.extract_semantic(target_16k)

        mel_source = self.mel_fn(source_tensor)
        mel_target = self.mel_fn(target_tensor)

        style = self.compute_style(target_16k)

        F0_ori_np = self.f0_extractor.infer_from_audio(target_16k[0], thred=0.03)
        F0_alt_np = self.f0_extractor.infer_from_audio(source_16k[0], thred=0.03)

        if device.type == "mps":
            F0_ori = torch.from_numpy(F0_ori_np).float().to(device)[None]
            F0_alt = torch.from_numpy(F0_alt_np).float().to(device)[None]
        else:
            F0_ori = torch.from_numpy(F0_ori_np).to(device)[None]
            F0_alt = torch.from_numpy(F0_alt_np).to(device)[None]

        # 3.3 — F0 adjustment and length regulation
        shifted_f0_alt = self.adjust_f0(F0_alt, F0_ori, auto_f0_adjust, pitch_shift)

        target_lengths = torch.LongTensor([int(mel_source.size(2) * length_adjust)]).to(
            device
        )
        target2_lengths = torch.LongTensor([mel_target.size(2)]).to(device)

        cond, _, _codes, _cl, _cb = self.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
        )
        prompt_condition, _, _codes, _cl, _cb = self.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )

        # 3.4 — chunked CFM inference loop
        max_context_window = self.sr // self.hop_size * self.dit_max_context_len
        max_source_window = max_context_window - mel_target.size(2)
        overlap_wave_len = self.overlap_frame_len * self.hop_size

        processed_frames = 0
        generated_wave_chunks = []
        previous_chunk = None

        while processed_frames < cond.size(1):
            chunk_cond = cond[
                :, processed_frames : processed_frames + max_source_window
            ]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

            with torch.autocast(device_type=device.type, dtype=dtype):
                vc_target = self.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(device),
                    mel_target,
                    style,
                    None,
                    diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate,
                )
                vc_target = vc_target[:, :, mel_target.size(-1) :]

            vc_wave = self.vocoder(vc_target.float()).squeeze().cpu()
            if vc_wave.ndim == 1:
                vc_wave = vc_wave.unsqueeze(0)

            if processed_frames == 0:
                if is_last_chunk:
                    generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                    break
                generated_wave_chunks.append(
                    vc_wave[0, :-overlap_wave_len].cpu().numpy()
                )
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len
            elif is_last_chunk:
                output_wave = self.crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0].cpu().numpy(),
                    overlap_wave_len,
                )
                generated_wave_chunks.append(output_wave)
                processed_frames += vc_target.size(2) - self.overlap_frame_len
                break
            else:
                output_wave = self.crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                    overlap_wave_len,
                )
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len

        if not generated_wave_chunks:
            return np.array([])
        return np.concatenate(generated_wave_chunks)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str,
        checkpoint_path: str,
        device: torch.device,
        fp16: bool = False,
    ) -> VoiceConversionWrapper:
        import yaml
        from seed_vc.modules.audio import mel_spectrogram
        from seed_vc.modules.campplus.DTDNN import CAMPPlus
        from seed_vc.modules.commons import (
            build_model,
            load_checkpoint,
            recursive_munch,
        )
        from seed_vc.modules.rmvpe import RMVPE
        from seed_vc.utils.hf_utils import load_custom_model_from_hf

        if checkpoint_path is None:
            checkpoint_path = load_custom_model_from_hf(
                DEFAULT_CHECKPOINT_REPO_ID, DEFAULT_CHECKPOINT
            )
        if config_path is None:
            config_path = load_custom_model_from_hf(
                DEFAULT_CONFIG_REPO_ID, DEFAULT_CONFIG
            )

        # 4.1 — build model and load checkpoint
        config = yaml.safe_load(open(config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = "DiT"
        nets = build_model(model_params, stage="DiT")

        load_checkpoint(
            {"cfm": nets.cfm, "length_regulator": nets.length_regulator},
            None,
            checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        nets.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        nets.cfm.to(device)
        nets.length_regulator.to(device)

        # 4.2 — Whisper
        from transformers import AutoFeatureExtractor, WhisperModel

        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(
            whisper_name,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).to(device)
        del whisper_model.decoder
        whisper_model.eval()
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        # 4.3 — CAMPPlus
        campplus_ckpt = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
        campplus.eval().to(device)

        # 4.4 — RMVPE
        rmvpe_path = load_custom_model_from_hf(
            "lj1995/VoiceConversionWebUI", "rmvpe.pt", None
        )
        rmvpe = RMVPE(rmvpe_path, is_half=False, device=device)

        # 4.5 — vocoder
        vocoder_type = model_params.vocoder.type
        if vocoder_type == "bigvgan":
            from seed_vc.modules.bigvgan import bigvgan

            vocoder = bigvgan.BigVGAN.from_pretrained(
                model_params.vocoder.name, use_cuda_kernel=False
            )
            vocoder.remove_weight_norm()
            vocoder = vocoder.eval().to(device)
        elif vocoder_type == "hifigan":
            from seed_vc.modules.hifigan.f0_predictor import ConvRNNF0Predictor
            from seed_vc.modules.hifigan.generator import HiFTGenerator

            hift_config = yaml.safe_load(open("configs/hifigan.yml", "r"))
            vocoder = HiFTGenerator(
                **hift_config["hift"],
                f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]),
            )
            hift_path = load_custom_model_from_hf(
                "FunAudioLLM/CosyVoice-300M", "hift.pt", None
            )
            vocoder.load_state_dict(torch.load(hift_path, map_location="cpu"))
            vocoder.eval().to(device)
        elif vocoder_type == "vocos":
            vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, "r"))
            vocos_model_params = recursive_munch(vocos_config["model_params"])
            vocos_nets = build_model(vocos_model_params, stage="mel_vocos")
            load_checkpoint(
                vocos_nets,
                None,
                model_params.vocoder.vocos.path,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False,
            )
            for key in vocos_nets:
                vocos_nets[key].eval().to(device)
            vocoder = vocos_nets.decoder
        else:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")

        # 4.6 — mel function and wrapper assembly
        spect = config["preprocess_params"]["spect_params"]
        sr = config["preprocess_params"]["sr"]
        hop_size = spect["hop_length"]
        fmax_raw = spect.get("fmax", "None")
        fmax = None if str(fmax_raw) == "None" else 8000

        def mel_fn(x: torch.Tensor) -> torch.Tensor:
            return mel_spectrogram(
                x,
                n_fft=spect["n_fft"],
                num_mels=spect["n_mels"],
                sampling_rate=sr,
                hop_size=spect["hop_length"],
                win_size=spect["win_length"],
                fmin=spect.get("fmin", 0),
                fmax=fmax,
                center=False,
            )

        return cls(
            sr=sr,
            hop_size=hop_size,
            mel_fn=mel_fn,
            cfm=nets.cfm,
            length_regulator=nets.length_regulator,
            whisper_model=whisper_model,
            whisper_feature_extractor=whisper_feature_extractor,
            style_encoder=campplus,
            f0_extractor=rmvpe,
            vocoder=vocoder,
        )
