"""PatchCore model: build, fit the memory bank, and score (PLAN.md section 5.4, Phase 2 step 2).

This is a thin layer over anomalib's ``PatchcoreModel``, a plain PyTorch ``nn.Module``. It does not
use anomalib's ``Engine``, data modules, pre-processor or post-processor: those are not needed and
their defaults use test data for thresholds (PLAN.md section 5.4). Images arrive already prepared
by ``defect_core.preprocessing``; the threshold comes from our own calibration, not from here.

Call order (verified on anomalib 2.7.0, PLAN.md step 0.2):

1. ``build_model``: create the module.
2. ``fit_memory_bank``: a train-mode pass over the ``fit`` images collects patch features, then
   coreset subsampling builds the memory bank. The model is left in eval mode.
3. ``infer``: an eval-mode pass returns the raw image score and the anomaly map.

Saving and loading a fitted model is a separate step (``artifacts.py``, Phase 2 step 3).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


@dataclass(frozen=True)
class ModelConfig:
    """Frozen description of the model. Defaults are the primary candidate from PLAN.md 5.4."""

    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    num_neighbors: int = 9
    coreset_ratio: float = 0.1

    def to_dict(self) -> dict[str, object]:
        """Plain-data form for the manifest."""
        return {
            "backbone": self.backbone,
            "layers": list(self.layers),
            "num_neighbors": self.num_neighbors,
            "coreset_ratio": self.coreset_ratio,
        }


@dataclass(frozen=True)
class FitStats:
    """What a memory-bank fit did: counts, the memory bank's shape, and the time of each stage."""

    patch_features: int  # patch features collected before subsampling
    memory_bank_shape: tuple[int, int]
    features_seconds: float  # train-mode pass over the images
    coreset_seconds: float  # coreset subsampling


def build_model(config: ModelConfig, *, pre_trained: bool) -> PatchcoreModel:
    """Create a ``PatchcoreModel`` in eval mode with an empty memory bank.

    ``pre_trained`` has no default on purpose. ``True`` downloads ImageNet backbone weights (used
    when fitting). ``False`` makes no network access and leaves random weights (used when the
    weights are loaded from a saved artifact, and in tests).
    """
    model = PatchcoreModel(
        backbone=config.backbone,
        pre_trained=pre_trained,
        layers=list(config.layers),
        num_neighbors=config.num_neighbors,
    )
    return model.eval()


def fit_memory_bank(
    model: PatchcoreModel,
    batches: Iterable[torch.Tensor],
    coreset_ratio: float,
    seed: int,
) -> FitStats:
    """Build the memory bank from normal images and leave the model in eval mode.

    ``batches`` yields preprocessed tensors of shape (B, 3, H, W) (see ``preprocessing``). Both
    ``torch`` and ``numpy`` are seeded first, because the coreset sampler uses a random projection
    that torch's seed alone does not control (PLAN.md 9.3). Returns ``FitStats`` (patch count,
    memory bank shape, seconds per stage). The backbone weights are not changed by fitting.
    """
    if not 0 < coreset_ratio <= 1:
        raise ValueError(f"coreset_ratio must be in (0, 1], got {coreset_ratio}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model.embedding_store.clear()  # drop leftovers from an earlier, interrupted fit
    model.train()
    collected = 0
    try:
        started = time.perf_counter()
        with torch.no_grad():
            for batch in batches:
                _check_batch(batch)
                collected += model(batch).shape[0]
        if collected == 0:
            raise ValueError("fit_memory_bank received no images")
        features_seconds = time.perf_counter() - started
        started = time.perf_counter()
        model.subsample_embedding(sampling_ratio=coreset_ratio)
        coreset_seconds = time.perf_counter() - started
    except BaseException:
        model.embedding_store.clear()
        raise
    finally:
        model.eval()
    rows, width = model.memory_bank.shape
    return FitStats(collected, (int(rows), int(width)), features_seconds, coreset_seconds)


def infer(model: PatchcoreModel, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a batch. Returns the raw image scores (B,) and anomaly maps (B, 1, H, W).

    The scores are anomaly scores, not probabilities. The model must be in eval mode with a
    filled memory bank; an empty bank raises ``ValueError`` (from anomalib).
    """
    if model.training:
        raise RuntimeError("infer needs the model in eval mode; call model.eval() first")
    _check_batch(batch)
    with torch.no_grad():
        output = model(batch)
    return output.pred_score, output.anomaly_map


def _check_batch(batch: torch.Tensor) -> None:
    if not isinstance(batch, torch.Tensor) or batch.ndim != 4 or batch.shape[1] != 3:
        raise ValueError("expected a tensor of shape (B, 3, H, W)")
    if not batch.is_floating_point():
        raise ValueError(f"expected a floating-point tensor, got {batch.dtype}")
