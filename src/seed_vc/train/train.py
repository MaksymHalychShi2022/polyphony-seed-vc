import argparse
import glob
import os
import shutil
import sys
from typing import Any, Sequence

import torch
import torch.multiprocessing as mp
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from seed_vc.modules.commons import build_model, load_checkpoint, recursive_munch
from seed_vc.train.features_dataset import build_features_dataloader
from seed_vc.train.optimizers import build_optimizer
from seed_vc.utils.hf_utils import load_custom_model_from_hf

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"


class Trainer:
    def __init__(
        self,
        config_path: str,
        pretrained_ckpt_path: str | None,
        train_data_path: str,
        val_data_path: str | None,
        cache_root: str,
        run_name: str,
        batch_size: int = 0,
        num_workers: int = 0,
        steps: int = 1000,
        save_interval: int = 500,
        max_epochs: int = 1000,
        eval_interval: int = 1,
        require_cache: bool = True,
        device: str = "cuda:0",
    ) -> None:
        self.device = device
        config = yaml.safe_load(open(config_path))
        self.log_dir = os.path.join(config["log_dir"], run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        # copy config file to log dir
        shutil.copyfile(
            config_path, os.path.join(self.log_dir, os.path.basename(config_path))
        )
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
            data_path=train_data_path,
            spect_params=preprocess_params["spect_params"],
            whisper_model_name=whisper_model_name,
            cache_root=cache_root,
            sr=self.sr,
            batch_size=batch_size,
            num_workers=num_workers,
            require_cache=require_cache,
        )
        if val_data_path:
            try:
                self.val_dataloader = build_features_dataloader(
                    data_path=val_data_path,
                    spect_params=preprocess_params["spect_params"],
                    whisper_model_name=whisper_model_name,
                    cache_root=cache_root,
                    sr=self.sr,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    require_cache=require_cache,
                )
                self.has_val = True
            except ValueError:
                print(
                    "No valid validation data found in validation CSV; evaluation will be skipped."
                )
                self.val_dataloader = None
                self.has_val = False
        else:
            print("No validation dataset provided; evaluation will be skipped.")
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

        self.model_params = recursive_munch(config["model_params"])
        self.model = build_model(self.model_params, stage="DiT")

        _ = [self.model[key].to(device) for key in self.model]
        self.model.cfm.estimator.setup_caches(
            max_batch_size=batch_size, max_seq_length=8192
        )

        # initialize optimizers after preparing models for compatibility with FSDP
        self.optimizer = build_optimizer(
            {key: self.model[key] for key in self.model},
            lr=float(scheduler_params["base_lr"]),
        )

        if pretrained_ckpt_path is None:
            # find latest checkpoint
            available_checkpoints = glob.glob(
                os.path.join(self.log_dir, "DiT_epoch_*_step_*.pth")
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
                # delete the earliest checkpoint if we have more than 2
                if (
                    earliest_checkpoint != latest_checkpoint
                    and len(available_checkpoints) > 2
                ):
                    os.remove(earliest_checkpoint)
                    print(f"Removed {earliest_checkpoint}")
            elif config.get("pretrained_model", ""):
                latest_checkpoint = load_custom_model_from_hf(
                    "Plachta/Seed-VC", config["pretrained_model"], None
                )
            else:
                latest_checkpoint = ""
        else:
            assert os.path.exists(pretrained_ckpt_path), (
                f"Pretrained checkpoint {pretrained_ckpt_path} not found"
            )
            latest_checkpoint = pretrained_ckpt_path

        if os.path.exists(latest_checkpoint):
            self.model, self.optimizer, self.epoch, self.iters = load_checkpoint(
                self.model,
                self.optimizer,
                latest_checkpoint,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False,
            )
            print(f"Loaded checkpoint from {latest_checkpoint}")
        else:
            self.epoch, self.iters = 0, 0
            print("Failed to load any checkpoint, training from scratch.")

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
        _ = [self.model[key].train() for key in self.model]
        for i, batch in enumerate(tqdm(self.train_dataloader)):
            batch = self.move_batch_to_device(batch)
            loss = self.train_one_step(batch)
            self.ema_loss = (
                self.ema_loss * self.loss_smoothing_rate
                + loss * (1 - self.loss_smoothing_rate)
                if self.iters > 0
                else loss
            )
            if self.iters % self.log_interval == 0:
                self.writer.add_scalar("train/loss", self.ema_loss, self.iters)
                self.writer.flush()
            self.iters += 1

            if self.iters >= self.max_steps:
                break

            if self.iters % self.save_interval == 0:
                print("Saving..")
                state = {
                    "net": {key: self.model[key].state_dict() for key in self.model},
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.optimizer.scheduler_state_dict(),
                    "iters": self.iters,
                    "epoch": self.epoch,
                }
                save_path = os.path.join(
                    self.log_dir,
                    f"DiT_epoch_{self.epoch:05d}_step_{self.iters:05d}.pth",
                )
                torch.save(state, save_path)

                # find all checkpoints and remove old ones
                checkpoints = glob.glob(os.path.join(self.log_dir, "DiT_epoch_*.pth"))
                if len(checkpoints) > 2:
                    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    for cp in checkpoints[:-2]:
                        os.remove(cp)

    def evaluate(self) -> float | None:
        if not self.has_val or self.val_dataloader is None:
            return None
        _ = [self.model[key].eval() for key in self.model]
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
            self.writer.add_scalar("eval/loss", avg_loss, self.iters)
            self.writer.flush()
        _ = [self.model[key].train() for key in self.model]
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
                    print(f"Eval loss at epoch {epoch + 1}: {eval_loss:.4f}")
            if self.iters >= self.max_steps:
                break

        print("Saving final model..")
        state = {
            "net": {key: self.model[key].state_dict() for key in self.model},
        }
        os.makedirs(self.log_dir, exist_ok=True)
        save_path = os.path.join(self.log_dir, "ft_model.pth")
        torch.save(state, save_path)
        self.writer.close()
        print(f"Final model saved at {save_path}")


def main(args: argparse.Namespace) -> None:
    train_data_path = args.train_dataset or args.dataset
    if not train_data_path:
        raise ValueError("Provide --train-dataset (or legacy --dataset) for training.")

    trainer = Trainer(
        config_path=args.config,
        pretrained_ckpt_path=args.pretrained_ckpt,
        train_data_path=train_data_path,
        val_data_path=args.val_dataset,
        cache_root=args.cache_root,
        run_name=args.run_name,
        batch_size=args.batch_size,
        steps=args.max_steps,
        max_epochs=args.max_epochs,
        save_interval=args.save_every,
        num_workers=args.num_workers,
        eval_interval=args.eval_every,
        require_cache=args.require_cache,
        device=args.device,
    )
    trainer.train()


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
        "--train-dataset",
        type=str,
        default=None,
        help="Path to training CSV with source,target rows.",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default=None,
        help="Optional path to validation CSV with source,target rows.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Legacy alias for --train-dataset.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=".cache",
        help="Root cache directory. Features are loaded from <cache-root>/features/.",
    )
    parser.add_argument(
        "--require-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require all feature cache files to exist. "
            "Use --no-require-cache to allow computing missing features on the fly."
        ),
    )
    parser.add_argument("--run-name", type=str, default="my_run")
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
