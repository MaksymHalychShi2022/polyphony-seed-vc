import argparse
import glob
import os
import sys
from typing import Any, Sequence

import torch
import torch.multiprocessing as mp
import yaml

from seed_vc.train.features_dataset import build_features_dataloader
from seed_vc.train.logger import TrainLogger
from seed_vc.train.optimizers import build_optimizer
from seed_vc.train.seed_vc_model import SeedVCModel
from seed_vc.utils.hf_utils import load_custom_model_from_hf

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"


class Trainer:
    def __init__(
        self,
        model: SeedVCModel,
        config_path: str,
        logger: TrainLogger,
        batch_size: int = 0,
        num_workers: int = 0,
        steps: int = 1000,
        save_interval: int = 500,
        max_epochs: int = 1000,
        eval_interval: int = 1,
        require_features: bool = True,
        device: str = "cuda:0",
    ) -> None:
        self.device = device
        self.logger = logger
        config = yaml.safe_load(open(config_path))
        self.log_dir = str(self.logger.experiment_dir)
        batch_size = config.get("batch_size", 10) if batch_size == 0 else batch_size
        self.max_steps = steps

        self.n_epochs = max_epochs
        self.log_interval = config.get("log_interval", 10)
        self.save_interval = save_interval
        self.eval_interval = max(1, eval_interval)

        self.sr = config["preprocess_params"].get("sr", 22050)
        self.hop_length = config["preprocess_params"]["spect_params"].get(
            "hop_length", 256
        )
        self.win_length = config["preprocess_params"]["spect_params"].get(
            "win_length", 1024
        )
        self.n_fft = config["preprocess_params"]["spect_params"].get("n_fft", 1024)
        preprocess_params = config["preprocess_params"]
        speech_tokenizer = config["model_params"]["speech_tokenizer"]
        speech_tokenizer_type = speech_tokenizer.get("type", "cosyvoice")
        if speech_tokenizer_type != "whisper":
            raise ValueError(
                f"Unsupported speech tokenizer type: {speech_tokenizer_type}. Expected 'whisper'."
            )
        whisper_model_name = speech_tokenizer["name"]

        self.train_dataloader = build_features_dataloader(
            split="train",
            spect_params=preprocess_params["spect_params"],
            whisper_model_name=whisper_model_name,
            sr=self.sr,
            batch_size=batch_size,
            num_workers=num_workers,
            require_features=require_features,
        )
        try:
            self.val_dataloader = build_features_dataloader(
                split="val",
                spect_params=preprocess_params["spect_params"],
                whisper_model_name=whisper_model_name,
                sr=self.sr,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                require_features=require_features,
            )
            self.has_val = True
        except (ValueError, FileNotFoundError):
            self.logger.warning(
                "No valid validation data found; evaluation will be skipped."
            )
            self.val_dataloader = None
            self.has_val = False
        assert config["model_params"]["DiT"].get("f0_condition", False), (
            "Singing VC training requires model_params.DiT.f0_condition=true"
        )
        assert config["model_params"]["length_regulator"].get("f0_condition", False), (
            "Singing VC training requires model_params.length_regulator.f0_condition=true"
        )

        scheduler_params = {
            "warmup_steps": 0,
            "base_lr": 0.00001,
        }

        self.model = model.to(device)
        self.model.setup_caches(max_batch_size=batch_size, max_seq_length=8192)

        # initialize optimizers after preparing models for compatibility with FSDP
        self.optimizer = build_optimizer(
            self.model.components(),
            lr=float(scheduler_params["base_lr"]),
        )
        self.epoch, self.iters = 0, 0

    def train_one_step(self, batch: Sequence[Any], update: bool = True) -> float:
        (
            _src_mels,
            _src_mel_lengths,
            tgt_mels,
            tgt_mel_lengths,
            src_semantics,
            _src_semantic_lengths,
            tgt_semantics,
            _tgt_semantic_lengths,
            src_f0s,
            _src_f0_lengths,
            tgt_f0s,
            _tgt_f0_lengths,
            tgt_embeddings,
            _src_paths,
            _tgt_paths,
        ) = batch

        B = tgt_mels.size(0)
        target_size = tgt_mels.size(2)
        target = tgt_mels
        target_lengths = tgt_mel_lengths

        # interpolate speech token to match acoustic feature length
        alt_cond, _, _, alt_commitment_loss, alt_codebook_loss = (
            self.model.length_regulator(
                tgt_semantics,
                ylens=target_lengths,
                f0=tgt_f0s,
            )
        )
        ori_cond, _, _, ori_commitment_loss, ori_codebook_loss = (
            self.model.length_regulator(
                src_semantics,
                ylens=target_lengths,
                f0=src_f0s,
            )
        )
        if alt_commitment_loss is None:
            alt_commitment_loss = 0
            alt_codebook_loss = 0
            ori_commitment_loss = 0
            ori_codebook_loss = 0

        # randomly set a length as prompt
        prompt_len_max = target_lengths - 1
        prompt_len = (
            (torch.rand([B], device=alt_cond.device) * prompt_len_max).floor().long()
        )
        prompt_len[torch.rand([B], device=alt_cond.device) < 0.1] = 0

        # for prompt cond token, use ori_cond instead of alt_cond
        cond = alt_cond.clone()
        for bib in range(B):
            cond[bib, : prompt_len[bib]] = ori_cond[bib, : prompt_len[bib]]

        # diffusion target
        common_min_len = min(target_size, cond.size(1))
        target = target[:, :, :common_min_len]
        cond = cond[:, :common_min_len]
        target_lengths = torch.clamp(target_lengths, max=common_min_len)
        x = target

        y = tgt_embeddings.to(torch.float32)

        loss, _ = self.model.cfm(x, target_lengths, prompt_len, cond, y)

        loss_total = (
            loss
            + (alt_commitment_loss + ori_commitment_loss) * 0.05
            + (ori_codebook_loss + alt_codebook_loss) * 0.15
        )

        if update:
            self.optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.cfm.parameters(), 10.0)
            torch.nn.utils.clip_grad_norm_(
                self.model.length_regulator.parameters(), 10.0
            )
            self.optimizer.step("cfm")
            self.optimizer.step("length_regulator")
            self.optimizer.scheduler(key="cfm")
            self.optimizer.scheduler(key="length_regulator")

        return loss.detach().item()

    def move_batch_to_device(self, batch: Sequence[Any]) -> list[Any]:
        return [b.to(self.device) if torch.is_tensor(b) else b for b in batch]

    def train_one_epoch(self) -> None:
        self.model.train()
        self.logger.progress(
            total=len(self.train_dataloader),
            desc=f"Epoch {self.epoch + 1}",
            advance=0,
        )
        for batch in self.train_dataloader:
            batch = self.move_batch_to_device(batch)
            loss = self.train_one_step(batch)
            self.ema_loss = (
                self.ema_loss * self.loss_smoothing_rate
                + loss * (1 - self.loss_smoothing_rate)
                if self.iters > 0
                else loss
            )
            if self.iters % self.log_interval == 0:
                self.logger.metric("loss", self.ema_loss, self.iters, context="train")
            self.iters += 1
            self.logger.progress()

            if self.iters >= self.max_steps:
                break

            if self.iters % self.save_interval == 0:
                self.logger.info("Saving checkpoint")
                save_path = os.path.join(
                    self.log_dir,
                    f"DiT_epoch_{self.epoch:05d}_step_{self.iters:05d}.pth",
                )
                self.save_state(save_path)

                # find all checkpoints and remove old ones
                checkpoints = glob.glob(os.path.join(self.log_dir, "DiT_epoch_*.pth"))
                if len(checkpoints) > 2:
                    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    for cp in checkpoints[:-2]:
                        os.remove(cp)
        self.logger.progress(close=True)

    def evaluate(self) -> float | None:
        if not self.has_val or self.val_dataloader is None:
            return None
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self.move_batch_to_device(batch)
                loss = self.train_one_step(batch, update=False)
                total_loss += loss
                n_batches += 1
        avg_loss = (total_loss / n_batches) if n_batches > 0 else None
        if avg_loss is not None:
            self.logger.metric("loss", avg_loss, self.iters, context="eval")
        self.model.train()
        return avg_loss

    def train(self) -> None:
        self.ema_loss = 0
        self.loss_smoothing_rate = 0.99
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            self.train_one_epoch()
            if (epoch + 1) % self.eval_interval == 0 and self.has_val:
                eval_loss = self.evaluate()
                if eval_loss is not None:
                    self.logger.info(f"Eval loss at epoch {epoch + 1}: {eval_loss:.4f}")
            if self.iters >= self.max_steps:
                break

        self.logger.info("Saving final model")
        os.makedirs(self.log_dir, exist_ok=True)
        save_path = os.path.join(self.log_dir, "ft_model.pth")
        self.model.save_weights(save_path)
        self.logger.info(f"Final model saved at {save_path}")

    def save_state(self, path: str) -> None:
        state = {
            "net": self.model.state_dict_for_checkpoint(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.optimizer.scheduler_state_dict(),
            "iters": self.iters,
            "epoch": self.epoch,
        }
        torch.save(state, path)

    def load_state(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        checkpoint_net = state["net"] if "net" in state else state
        self.model.load_state_dict_from_checkpoint(
            checkpoint_net,
            ignore_modules=[],
            is_distributed=False,
        )
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            self.optimizer.load_scheduler_state_dict(state["scheduler"])
        self.iters = int(state.get("iters", 0))
        self.epoch = int(state.get("epoch", 0))


def main(args: argparse.Namespace) -> None:
    config = yaml.safe_load(open(args.config))
    logger = TrainLogger(experiment_name=args.run_name)
    logger.start()
    logger.save_artifact(args.config)

    model = SeedVCModel(config["model_params"])

    latest_checkpoint: str = ""
    if args.pretrained_ckpt is None:
        available_checkpoints = glob.glob(
            os.path.join(str(logger.experiment_dir), "DiT_epoch_*_step_*.pth")
        )
        if len(available_checkpoints) > 0:
            latest_checkpoint = max(
                available_checkpoints,
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            earliest_checkpoint = min(
                available_checkpoints,
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            if (
                earliest_checkpoint != latest_checkpoint
                and len(available_checkpoints) > 2
            ):
                os.remove(earliest_checkpoint)
                logger.info(f"Removed {earliest_checkpoint}")
        elif config.get("pretrained_model", ""):
            hf_checkpoint = load_custom_model_from_hf(
                "Plachta/Seed-VC", config["pretrained_model"], None
            )
            latest_checkpoint = (
                hf_checkpoint if isinstance(hf_checkpoint, str) else hf_checkpoint[0]
            )
        else:
            latest_checkpoint = ""
    else:
        assert os.path.exists(args.pretrained_ckpt), (
            f"Pretrained checkpoint {args.pretrained_ckpt} not found"
        )
        latest_checkpoint = str(args.pretrained_ckpt)

    if os.path.exists(str(latest_checkpoint)):
        model.load_weights(str(latest_checkpoint))
        logger.info(f"Loaded checkpoint from {latest_checkpoint}")
    else:
        logger.warning("Failed to load any checkpoint, training from scratch.")

    trainer = Trainer(
        model=model,
        config_path=args.config,
        logger=logger,
        batch_size=args.batch_size,
        steps=args.max_steps,
        max_epochs=args.max_epochs,
        save_interval=args.save_every,
        num_workers=args.num_workers,
        eval_interval=args.eval_every,
        require_features=args.require_features,
        device=args.device,
    )
    try:
        trainer.train()
    except Exception as exc:
        logger.error("Training failed", exc=exc)
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        mp.freeze_support()
        mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    )
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument(
        "--require-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require all feature files to exist (DATA_FEATURES). "
            "Use --no-require-features to compute missing features on the fly."
        ),
    )
    parser.add_argument("--run-name", type=str, default="hahahah")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run evaluation every N epochs.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    main(args)
