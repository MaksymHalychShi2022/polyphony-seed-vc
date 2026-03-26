import glob
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from seed_vc.train.features_dataset import build_features_dataloader
from seed_vc.train.optimizers import build_optimizer
from seed_vc.train.seed_vc_model import SeedVCModel
from seed_vc.utils.hf_utils import load_custom_model_from_hf

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

log = logging.getLogger(__name__)

_CONFIGS_DIR = str(Path(__file__).parent.parent.parent.parent / "configs")


class Trainer:
    def __init__(
        self,
        model: SeedVCModel,
        config: dict,
        log_dir: str,
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
        self.log_dir = log_dir
        self._setup_loggers(config)
        batch_size = config.get("batch_size", 10) if batch_size == 0 else batch_size
        self.max_steps = steps

        self.n_epochs = max_epochs
        self.log_interval = config.get("log_interval", 10)
        self.save_interval = save_interval
        self.eval_interval = max(1, eval_interval)

        preprocess_params = config["preprocess_params"]
        self.sr = preprocess_params.get("sr", 22050)
        speech_tokenizer = config["model_params"]["speech_tokenizer"]
        if speech_tokenizer.get("type", "cosyvoice") != "whisper":
            raise ValueError(
                f"Unsupported speech tokenizer type: {speech_tokenizer.get('type')}. Expected 'whisper'."
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
            log.warning("No valid validation data found; evaluation will be skipped.")
            self.val_dataloader = None
            self.has_val = False

        assert config["model_params"]["DiT"].get("f0_condition", False), (
            "Singing VC training requires model_params.DiT.f0_condition=true"
        )
        assert config["model_params"]["length_regulator"].get("f0_condition", False), (
            "Singing VC training requires model_params.length_regulator.f0_condition=true"
        )

        self.model = model.to(device)
        self.model.setup_caches(max_batch_size=batch_size, max_seq_length=8192)
        self.optimizer = build_optimizer(self.model.components(), lr=0.00001)
        self.epoch, self.iters = 0, 0

    def _setup_loggers(self, config: dict) -> None:
        self._tb = (
            SummaryWriter(self.log_dir)
            if config.get("tensorboard_logging", True)
            else None
        )
        self._mlflow = False
        if config.get("mlflow_logging", False):
            uri = os.getenv("MLFLOW_TRACKING_URI")
            if uri:
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(config["run_name"])
                mlflow.start_run(run_name=config["run_name"])
                self._mlflow = True
            else:
                log.warning(
                    "mlflow_logging=true but MLFLOW_TRACKING_URI is not set, skipping."
                )

    def log_metric(self, name: str, value: float, step: int) -> None:
        if self._tb is not None:
            self._tb.add_scalar(name, value, step)
            self._tb.flush()
        if self._mlflow:
            mlflow.log_metric(name, float(value), step=step)

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
        target_lengths = tgt_mel_lengths

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
            alt_commitment_loss = ori_commitment_loss = alt_codebook_loss = (
                ori_codebook_loss
            ) = 0

        prompt_len_max = target_lengths - 1
        prompt_len = (
            (torch.rand([B], device=alt_cond.device) * prompt_len_max).floor().long()
        )
        prompt_len[torch.rand([B], device=alt_cond.device) < 0.1] = 0

        cond = alt_cond.clone()
        for bib in range(B):
            cond[bib, : prompt_len[bib]] = ori_cond[bib, : prompt_len[bib]]

        common_min_len = min(target_size, cond.size(1))
        target = tgt_mels[:, :, :common_min_len]
        cond = cond[:, :common_min_len]
        target_lengths = torch.clamp(target_lengths, max=common_min_len)

        loss, _ = self.model.cfm(
            target, target_lengths, prompt_len, cond, tgt_embeddings.to(torch.float32)
        )
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
        for batch in tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}"):
            batch = self.move_batch_to_device(batch)
            loss = self.train_one_step(batch)
            self.ema_loss = (
                self.ema_loss * self.loss_smoothing_rate
                + loss * (1 - self.loss_smoothing_rate)
                if self.iters > 0
                else loss
            )
            if self.iters % self.log_interval == 0:
                self.log_metric("train/loss", self.ema_loss, self.iters)
            self.iters += 1

            if self.iters >= self.max_steps:
                break

            if self.iters % self.save_interval == 0:
                log.info("Saving checkpoint")
                save_path = os.path.join(
                    self.log_dir,
                    f"DiT_epoch_{self.epoch:05d}_step_{self.iters:05d}.pth",
                )
                self.save_state(save_path)
                checkpoints = glob.glob(os.path.join(self.log_dir, "DiT_epoch_*.pth"))
                if len(checkpoints) > 2:
                    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    for cp in checkpoints[:-2]:
                        os.remove(cp)

    def evaluate(self) -> float | None:
        if not self.has_val or self.val_dataloader is None:
            return None
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self.move_batch_to_device(batch)
                total_loss += self.train_one_step(batch, update=False)
                n_batches += 1
        avg_loss = (total_loss / n_batches) if n_batches > 0 else None
        if avg_loss is not None:
            self.log_metric("eval/loss", avg_loss, self.iters)
        self.model.train()
        return avg_loss

    def train(self) -> None:
        self.ema_loss = 0.0
        self.loss_smoothing_rate = 0.99
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            self.train_one_epoch()
            if (epoch + 1) % self.eval_interval == 0 and self.has_val:
                eval_loss = self.evaluate()
                if eval_loss is not None:
                    log.info(f"Eval loss at epoch {epoch + 1}: {eval_loss:.4f}")
            if self.iters >= self.max_steps:
                break

        log.info("Saving final model")
        os.makedirs(self.log_dir, exist_ok=True)
        save_path = os.path.join(self.log_dir, "ft_model.pth")
        self.model.save_weights(save_path)
        log.info(f"Final model saved at {save_path}")

        if self._tb is not None:
            self._tb.close()
        if self._mlflow:
            mlflow.end_run()

    def save_state(self, path: str) -> None:
        torch.save(
            {
                "net": self.model.state_dict_for_checkpoint(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.optimizer.scheduler_state_dict(),
                "iters": self.iters,
                "epoch": self.epoch,
            },
            path,
        )

    def load_state(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        checkpoint_net = state["net"] if "net" in state else state
        self.model.load_state_dict_from_checkpoint(
            checkpoint_net, ignore_modules=[], is_distributed=False
        )
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            self.optimizer.load_scheduler_state_dict(state["scheduler"])
        self.iters = int(state.get("iters", 0))
        self.epoch = int(state.get("epoch", 0))


@hydra.main(config_path=_CONFIGS_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig

    config: dict = OmegaConf.to_container(cfg, resolve=True)
    log_dir = HydraConfig.get().runtime.output_dir

    model = SeedVCModel(config["model_params"])

    pretrained_ckpt = config.get("pretrained_ckpt")
    if pretrained_ckpt:
        assert os.path.exists(pretrained_ckpt), (
            f"Pretrained checkpoint not found: {pretrained_ckpt}"
        )
        model.load_weights(pretrained_ckpt)
        log.info(f"Loaded checkpoint from {pretrained_ckpt}")
    elif config.get("pretrained_model", ""):
        hf_checkpoint = load_custom_model_from_hf(
            "Plachta/Seed-VC", config["pretrained_model"], None
        )
        ckpt_path = (
            hf_checkpoint if isinstance(hf_checkpoint, str) else hf_checkpoint[0]
        )
        model.load_weights(ckpt_path)
        log.info(f"Loaded HF checkpoint from {ckpt_path}")
    else:
        log.warning("No checkpoint provided, training from scratch.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        model=model,
        config=config,
        log_dir=log_dir,
        batch_size=config.get("batch_size", 2),
        steps=config.get("max_steps", 1000),
        max_epochs=config.get("max_epochs", 1000),
        save_interval=config.get("save_every", 100),
        num_workers=config.get("num_workers", 0),
        eval_interval=config.get("eval_every", 1),
        require_features=config.get("require_features", True),
        device=device,
    )
    try:
        trainer.train()
    except Exception:
        log.exception("Training failed")
        raise


if __name__ == "__main__":
    main()
