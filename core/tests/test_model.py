"""Tests for defect_core.model (Phase 2, step 2).

Tiny random images and a small randomly initialised backbone are used. They check the wiring and
the order of calls, not detection quality. The network is blocked, so any attempt to download
pretrained weights fails the test.
"""

from __future__ import annotations

import dataclasses
import socket

import pytest
import torch
from defect_core.model import ModelConfig, build_model, fit_memory_bank, infer

SIZE = 128
CONFIG = ModelConfig(backbone="resnet18")


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network access attempted during test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


def _images(n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, SIZE, SIZE, generator=generator)


def _fitted(seed: int = 0):
    torch.manual_seed(123)
    model = build_model(CONFIG, pre_trained=False)
    fit_memory_bank(model, [_images(4, 1), _images(4, 2)], CONFIG.coreset_ratio, seed)
    return model


def test_config_defaults_are_the_primary_candidate() -> None:
    config = ModelConfig()
    assert config.backbone == "wide_resnet50_2"
    assert config.layers == ("layer2", "layer3")
    assert config.num_neighbors == 9
    assert config.coreset_ratio == 0.1
    assert config.to_dict() == {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "num_neighbors": 9,
        "coreset_ratio": 0.1,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.backbone = "resnet18"  # type: ignore[misc]


def test_pre_trained_must_be_chosen_explicitly() -> None:
    with pytest.raises(TypeError):
        build_model(CONFIG)  # type: ignore[call-arg]


def test_build_makes_no_network_call_and_starts_empty_in_eval_mode() -> None:
    model = build_model(CONFIG, pre_trained=False)
    assert not model.training
    assert model.memory_bank.numel() == 0


def test_scoring_with_an_empty_bank_is_rejected() -> None:
    model = build_model(CONFIG, pre_trained=False)
    with pytest.raises(ValueError, match="Memory bank is empty"):
        infer(model, _images(1, 0))


def test_fit_builds_a_smaller_bank_and_leaves_eval_mode() -> None:
    model = build_model(CONFIG, pre_trained=False)
    collected = fit_memory_bank(model, [_images(4, 1), _images(2, 2)], 0.1, seed=0)
    assert not model.training
    assert model.embedding_store == []
    # 6 images were used; each gives (SIZE/8)^2 patch features (layer2 has stride 8)
    assert collected == 6 * (SIZE // 8) ** 2
    assert model.memory_bank.shape[0] == pytest.approx(0.1 * collected, abs=1)


def test_infer_returns_scores_and_maps_with_expected_shapes() -> None:
    model = _fitted()
    score, anomaly_map = infer(model, _images(3, 9))
    assert score.shape == (3,)
    assert anomaly_map.shape == (3, 1, SIZE, SIZE)
    assert torch.isfinite(score).all() and torch.isfinite(anomaly_map).all()


def test_an_image_in_the_bank_scores_lower_than_a_very_different_image() -> None:
    torch.manual_seed(123)
    model = build_model(CONFIG, pre_trained=False)
    normal = _images(6, 1)
    fit_memory_bank(model, [normal], 1.0, seed=0)  # keep every feature
    seen, _ = infer(model, normal[:1])
    odd, _ = infer(model, torch.full((1, 3, SIZE, SIZE), 5.0))
    assert float(seen) < float(odd)


def test_same_seed_gives_identical_banks_and_scores() -> None:
    a, b = _fitted(seed=0), _fitted(seed=0)
    assert torch.equal(a.memory_bank, b.memory_bank)
    sa, ma = infer(a, _images(2, 5))
    sb, mb = infer(b, _images(2, 5))
    assert torch.equal(sa, sb) and torch.equal(ma, mb)


def test_fitting_does_not_change_backbone_weights_or_batchnorm_statistics() -> None:
    torch.manual_seed(123)
    model = build_model(CONFIG, pre_trained=False)
    before = {k: v.clone() for k, v in model.state_dict().items() if k != "memory_bank"}
    fit_memory_bank(model, [_images(4, 1)], 0.1, seed=0)
    after = {k: v for k, v in model.state_dict().items() if k != "memory_bank"}
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_a_second_fit_replaces_the_first() -> None:
    model = build_model(CONFIG, pre_trained=False)
    fit_memory_bank(model, [_images(4, 1)], 0.5, seed=0)
    first = model.memory_bank.clone()
    fit_memory_bank(model, [_images(4, 2)], 0.5, seed=0)
    assert model.embedding_store == []
    assert model.memory_bank.shape == first.shape
    assert not torch.equal(model.memory_bank, first)


def test_bad_input_is_refused_and_a_failed_fit_leaves_no_leftovers() -> None:
    model = build_model(CONFIG, pre_trained=False)
    with pytest.raises(ValueError, match="no images"):
        fit_memory_bank(model, [], 0.1, seed=0)
    with pytest.raises(ValueError, match="coreset_ratio"):
        fit_memory_bank(model, [_images(1, 0)], 0.0, seed=0)
    with pytest.raises(ValueError, match="coreset_ratio"):
        fit_memory_bank(model, [_images(1, 0)], 1.5, seed=0)
    with pytest.raises(ValueError, match=r"\(B, 3, H, W\)"):
        fit_memory_bank(model, [_images(2, 0), torch.rand(3, SIZE, SIZE)], 0.1, seed=0)
    assert model.embedding_store == [] and not model.training
    with pytest.raises(ValueError, match="floating-point"):
        infer(_fitted(), torch.zeros(1, 3, SIZE, SIZE, dtype=torch.uint8))


def test_infer_refuses_train_mode() -> None:
    model = _fitted()
    model.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        infer(model, _images(1, 0))
