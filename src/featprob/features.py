"""Penultimate-layer feature extraction for PyTorch models.

``torch`` is imported lazily so that ``import featprob.features`` works in an
environment without the ``torch`` extra.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

_TORCH_MSG = "torch is required for feature extraction; install with `uv sync --extra torch`"


def _import_torch() -> Any:
    """Import torch on demand.

    Returns:
        The ``torch`` module.

    Raises:
        ImportError: If torch is not installed.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_TORCH_MSG) from exc
    return torch


class FeatureExtractor:
    """Capture the input of the final ``Linear`` layer (the penultimate features).

    By default the last ``torch.nn.Linear`` module found is hooked with a forward
    pre-hook, so the captured tensor is exactly what the classifier head sees.
    A specific module can be hooked by name via ``layer_name`` (its *output* is
    captured). 4-D outputs (N, C, H, W) are globally average-pooled.
    """

    def __init__(self, model: Any, layer_name: str | None = None, device: str = "cpu") -> None:
        """Register the hook.

        Args:
            model: A ``torch.nn.Module``.
            layer_name: Name from ``model.named_modules()``; ``None`` selects the
                input of the last ``Linear`` layer.
            device: Device on which inputs are evaluated.

        Raises:
            ValueError: If the layer cannot be found.
        """
        torch = _import_torch()
        self.model = model.to(device).eval()
        self.device = device
        self._captured: Any = None
        modules = dict(self.model.named_modules())
        if layer_name is None:
            linears = [m for m in modules.values() if isinstance(m, torch.nn.Linear)]
            if not linears:
                raise ValueError("model has no Linear layer; pass layer_name explicitly")
            self._handle = linears[-1].register_forward_pre_hook(self._pre_hook)
        else:
            if layer_name not in modules:
                raise ValueError(f"layer {layer_name!r} not found in model")
            self._handle = modules[layer_name].register_forward_hook(self._hook)

    def _pre_hook(self, _module: Any, args: tuple[Any, ...]) -> None:
        self._captured = args[0].detach()

    def _hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> None:
        self._captured = output.detach()

    def close(self) -> None:
        """Remove the registered hook."""
        self._handle.remove()

    def __enter__(self) -> FeatureExtractor:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Remove the hook on exit."""
        self.close()

    def extract_batch(self, inputs: Any) -> NDArray[np.float32]:
        """Run one batch and return flattened penultimate features.

        Args:
            inputs: A tensor of shape (N, ...).

        Returns:
            Array of shape (N, D).
        """
        torch = _import_torch()
        with torch.no_grad():
            self.model(inputs.to(self.device))
        feats = self._captured
        if feats is None:
            raise RuntimeError("hook did not fire; layer is not on the forward path")
        if feats.dim() > 2:
            feats = feats.flatten(2).mean(dim=2)
        return feats.reshape(feats.shape[0], -1).float().cpu().numpy()

    def extract(self, loader: Iterable[Any]) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Extract features and labels from an iterable of ``(inputs, labels)``.

        Args:
            loader: Typically a ``torch.utils.data.DataLoader`` (no shuffling needed).

        Returns:
            ``(features, labels)`` with shapes (N, D) and (N,).
        """
        feats: list[NDArray[np.float32]] = []
        labels: list[NDArray[np.int64]] = []
        for inputs, targets in loader:
            feats.append(self.extract_batch(inputs))
            labels.append(np.asarray(targets).astype(np.int64))
        return np.concatenate(feats), np.concatenate(labels)
