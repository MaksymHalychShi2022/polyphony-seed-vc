from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from seed_vc.modules.commons import build_model, recursive_munch


class SeedVCModel(nn.Module):
    def __init__(self, model_params: Mapping[str, Any]) -> None:
        super().__init__()
        self.model_params = recursive_munch(model_params)
        nets = build_model(self.model_params, stage="DiT")
        self.cfm = nets.cfm
        self.length_regulator = nets.length_regulator

    def components(self) -> dict[str, nn.Module]:
        return {
            "cfm": self.cfm,
            "length_regulator": self.length_regulator,
        }

    def setup_caches(self, max_batch_size: int, max_seq_length: int = 8192) -> None:
        self.cfm.estimator.setup_caches(
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
        )

    def state_dict_for_checkpoint(self) -> dict[str, Any]:
        return {key: module.state_dict() for key, module in self.components().items()}

    @staticmethod
    def _strip_prefix(
        state_dict: dict[str, Any], prefix: str = "module."
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                output[key[len(prefix) :]] = value
            else:
                output[key] = value
        return output

    def load_state_dict_from_checkpoint(
        self,
        checkpoint_net: Mapping[str, Any],
        ignore_modules: Sequence[str] | None = None,
        is_distributed: bool = False,
    ) -> None:
        ignored = set(ignore_modules or [])
        for key, module in self.components().items():
            if key not in checkpoint_net or key in ignored:
                continue
            module_state = dict(checkpoint_net[key])
            if not is_distributed:
                module_state = self._strip_prefix(module_state)
            current_state = module.state_dict()
            filtered_state = {
                name: tensor
                for name, tensor in module_state.items()
                if name in current_state and current_state[name].shape == tensor.shape
            }
            skipped_keys = set(module_state.keys()) - set(filtered_state.keys())
            if skipped_keys:
                print(
                    f"Warning: Skipped loading some keys for {key} due to shape mismatch: {skipped_keys}"
                )
            module.load_state_dict(filtered_state, strict=False)

    def save_weights(self, path: str) -> None:
        torch.save({"net": self.state_dict_for_checkpoint()}, path)

    def load_weights(
        self,
        path: str,
        ignore_modules: Sequence[str] | None = None,
        is_distributed: bool = False,
    ) -> None:
        state = torch.load(path, map_location="cpu")
        checkpoint_net = state["net"] if "net" in state else state
        self.load_state_dict_from_checkpoint(
            checkpoint_net,
            ignore_modules=ignore_modules,
            is_distributed=is_distributed,
        )
