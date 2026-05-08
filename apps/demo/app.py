import argparse
import os

import gradio as gr
import torch

from seed_vc.modules.commons import str2bool
from seed_vc.modules.vc_wrapper import VoiceConversionWrapper

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

fp16 = False
device = None
wrapper: VoiceConversionWrapper | None = None


@torch.inference_mode()
def voice_conversion(
    source,
    target,
    diffusion_steps,
    length_adjust,
    inference_cfg_rate,
    auto_f0_adjust,
    pitch_shift,
):
    audio = wrapper.convert(
        source_path=source,
        target_path=target,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=pitch_shift,
        device=device,
        dtype=torch.float16 if fp16 else torch.float32,
    )
    return wrapper.sr, audio


def main(args):
    global wrapper, fp16
    fp16 = args.fp16
    print(f"Using device: {device}")
    print(f"Using fp16: {fp16}")

    if args.checkpoint:
        print(f"Using custom checkpoint: {args.checkpoint}")

    wrapper = VoiceConversionWrapper.from_config(
        config_path=args.config or None,
        checkpoint_path=args.checkpoint or None,
        device=device,
        fp16=fp16,
    )

    with gr.Blocks(title="Seed Voice Conversion") as demo:
        gr.Markdown("# Seed Voice Conversion")
        with gr.Row():
            source_audio = gr.Audio(type="filepath", label="Source Audio")
            target_audio = gr.Audio(type="filepath", label="Reference Audio")
        with gr.Accordion("Advanced settings", open=False):
            diffusion_steps = gr.Slider(
                minimum=1,
                maximum=200,
                value=10,
                step=1,
                label="Diffusion Steps",
                info="10 by default, 50~100 for best quality",
            )
            length_adjust = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                step=0.1,
                value=1.0,
                label="Length Adjust",
                info="<1.0 for speed-up speech, >1.0 for slow-down speech",
            )
            inference_cfg_rate = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.7,
                label="Inference CFG Rate",
                info="has subtle influence",
            )
        convert_button = gr.Button("Convert")
        output_audio = gr.Audio(
            label="Output Audio",
            streaming=False,
            format="wav",
            type="numpy",
        )

        convert_button.click(
            lambda src, tgt, steps, adj, cfg: voice_conversion(
                src, tgt, steps, adj, cfg, True, 0
            ),
            inputs=[
                source_audio,
                target_audio,
                diffusion_steps,
                length_adjust,
                inference_cfg_rate,
            ],
            outputs=output_audio,
        )

    demo.launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, help="Path to the checkpoint file", default=None
    )
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default=None
    )
    parser.add_argument(
        "--share",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to share the app",
    )
    parser.add_argument(
        "--fp16",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use fp16",
        default=True,
    )
    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    args = parser.parse_args()
    cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda"

    if torch.cuda.is_available():
        device = torch.device(cuda_target)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    main(args)
