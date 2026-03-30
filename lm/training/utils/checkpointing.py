import os
import typing

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    meta: dict | None = None,
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "meta": meta or {},
    }
    torch.save(state, out)

    return


def read_checkpoint_meta(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    device: torch.device | None = None,
) -> dict:
    state = torch.load(src, weights_only=False, map_location=device)
    return state.get("meta") or {}


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> int:
    state = torch.load(src, weights_only=False, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])

    return state["iteration"]
