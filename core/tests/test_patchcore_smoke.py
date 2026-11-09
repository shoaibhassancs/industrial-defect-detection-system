"""PatchCore compatibility tests (Phase 0, step 0.3).

These check that the installed stack works: build, collect features, build the memory bank,
score. They use tiny random images and a small backbone, so they run in seconds.
They say nothing about detection quality, real-size memory cost, or saving/reloading.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

LAYERS = ["layer2", "layer3"]
NUM_IMAGES = 4
IMAGE_SIZE = 128


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to use the network (e.g. to download pretrained weights)."""

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network access attempted during test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


def build_small_model() -> PatchcoreModel:
    return PatchcoreModel(layers=LAYERS, backbone="resnet18", pre_trained=False)


def fit_and_score(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full tiny pipeline. Returns (embedding, memory_bank, pred_score, anomaly_map).

    Seeds BOTH torch and numpy: anomalib's coreset sampler uses a random projection that
    torch's seed alone does not control.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_small_model()
    train_images = torch.rand(NUM_IMAGES, 3, IMAGE_SIZE, IMAGE_SIZE)
    test_images = torch.rand(NUM_IMAGES, 3, IMAGE_SIZE, IMAGE_SIZE)

    model.train()
    with torch.no_grad():
        embedding = model(train_images)
    model.subsample_embedding(sampling_ratio=0.1)

    model.eval()
    with torch.no_grad():
        output = model(test_images)
    return embedding, model.memory_bank, output.pred_score, output.anomaly_map


def test_build_without_pretrained_weights_makes_no_network_calls() -> None:
    # The autouse fixture blocks the network; construction must still succeed.
    model = build_small_model()
    assert model.memory_bank.numel() == 0


def test_primary_backbone_name_is_accepted() -> None:
    model = PatchcoreModel(layers=LAYERS, backbone="wide_resnet50_2", pre_trained=False)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_eval_with_empty_memory_bank_is_rejected() -> None:
    model = build_small_model()
    model.eval()
    with pytest.raises(ValueError, match="Memory bank is empty"), torch.no_grad():
        model(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))


def test_fit_then_score_returns_expected_shapes() -> None:
    embedding, bank, score, anomaly_map = fit_and_score(seed=0)

    assert embedding.ndim == 2 and embedding.shape[0] > 0
    assert 0 < bank.shape[0] < embedding.shape[0]
    assert bank.shape[1] == embedding.shape[1]
    assert score.numel() == NUM_IMAGES
    assert tuple(anomaly_map.shape) == (NUM_IMAGES, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.isfinite(score).all()
    assert torch.isfinite(anomaly_map).all()


def test_same_seeds_give_identical_scores() -> None:
    _, _, score_a, map_a = fit_and_score(seed=0)
    _, _, score_b, map_b = fit_and_score(seed=0)

    assert torch.equal(score_a, score_b)
    assert torch.equal(map_a, map_b)
