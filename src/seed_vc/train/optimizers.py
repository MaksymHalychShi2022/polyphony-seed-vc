# coding:utf-8
from functools import reduce

import torch
from torch.optim import AdamW


class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(
            lambda x, y: x + y, [v.param_groups for v in self.optimizers.values()]
        )

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict()) for key in self.keys]
        return state_dicts

    def scheduler_state_dict(self):
        state_dicts = [(key, self.schedulers[key].state_dict()) for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except Exception:
                print("Unloaded %s" % key)

    def load_scheduler_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.schedulers[key].load_state_dict(val)
            except Exception:
                print("Unloaded %s" % key)

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step_batch(*args) for key in self.keys]


class MinLRExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-5):
        self.min_lr = min_lr
        super().__init__(optimizer, gamma)

    def get_lr(self):
        lrs = super().get_lr()
        return [max(lr, self.min_lr) for lr in lrs]


def build_optimizer(
    model_dict: dict, optimizer_cfg: dict, scheduler_cfg: dict
) -> MultiOptimizer:
    optim = {}
    for key, model in model_dict.items():
        optim[key] = AdamW(
            model.parameters(),
            lr=optimizer_cfg["lr"],
            betas=tuple(optimizer_cfg["betas"]),
            eps=optimizer_cfg["eps"],
            weight_decay=optimizer_cfg["weight_decay"],
        )

    schedulers = {
        key: MinLRExponentialLR(
            opt, gamma=scheduler_cfg["gamma"], min_lr=scheduler_cfg["min_lr"]
        )
        for key, opt in optim.items()
    }

    return MultiOptimizer(optim, schedulers)
