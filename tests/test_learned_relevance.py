"""Frozen-feature training, split isolation, checked artifacts and local runtime."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.image_ranking import ClipRanker, LearnedContextRanker, build_ranker, encoder_fingerprint
from semantic_image_stream.learned_relevance import (
    AdapterError, FEATURE_CHECKSUM, FEATURE_DIM, _auc, inspect_training_pack,
    load_adapter, pair_features, save_adapter, train_adapter,
)


@pytest.fixture(scope="module")
def training_data():
    random = np.random.default_rng(25)
    images = random.normal(size=(12, 512))
    texts = images + random.normal(scale=0.3, size=images.shape)
    return images, texts, [f"source-{index}" for index in range(12)], [
        "train"] * 4 + ["calibration"] * 4 + ["test"] * 4


@pytest.fixture(scope="module")
def trained(training_data):
    return train_adapter(*training_data, {"encoder_model": "openai/clip-vit-base-patch32",
                                          "encoder_fingerprint": "a" * 64,
                                          "dataset": "offline fixture", "license": "CC0"})


def test_pair_features_normalize_and_broadcast_with_exact_checksum():
    first = np.zeros(512)
    first[0] = 4
    second = np.zeros(512)
    second[1] = 2
    features = pair_features([first, second], first)
    assert features.shape == (2, 1025)
    assert features[0, 0] == features[0, -1] == 1
    assert np.sum(features[0, 512:1024]) == 0
    assert features[1, -1] == 0 and np.sum(features[1, 512:1024]) == 2
    assert len(FEATURE_CHECKSUM) == 64


def test_train_report_has_disjoint_counts_and_untouched_test_metrics(trained):
    artifact, report = trained
    assert artifact["report"] == report and report["train_ready"] is True
    assert len(artifact["weights"]) == FEATURE_DIM
    for values in report["counts"].values():
        assert values == {"images": 4, "groups": 4, "pairs": 12, "positives": 4, "negatives": 8}
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in report["held_out"].values())
    assert report["held_out"]["baseline_top1"] == report["held_out"]["baseline_pair_auc"] == 1
    assert artifact["recommended_threshold"] == report["calibration"]["threshold"]
    assert report["selection"]["split"] == "calibration"
    assert report["top1_direction"] == "text_to_image"
    assert "gain" not in report and "improved" not in report


def test_training_deterministic_and_test_values_cannot_affect_fit_or_calibration(training_data, trained):
    images, texts, groups, splits = training_data
    artifact, _ = train_adapter(*training_data, {"encoder_model": "openai/clip-vit-base-patch32"})
    np.testing.assert_allclose(artifact["weights"], trained[0]["weights"], rtol=0, atol=0)
    changed = texts.copy()
    changed[8:] = np.roll(changed[8:], 1, axis=0)
    altered, _ = train_adapter(images, changed, groups, splits, {"encoder_model": artifact["encoder_model"]})
    np.testing.assert_allclose(altered["weights"], artifact["weights"], rtol=0, atol=0)
    assert altered["bias"] == artifact["bias"] and altered["calibration"] == artifact["calibration"]
    assert altered["recommended_threshold"] == artifact["recommended_threshold"]
    assert altered["report"]["selection"] == artifact["report"]["selection"]
    assert altered["report"]["held_out"] != artifact["report"]["held_out"]


@pytest.mark.parametrize("bad", ["group_leak", "missing_split", "nonfinite", "zero_vector", "wrong_dim"])
def test_invalid_training_boundaries_fail_without_caption_or_path_leak(training_data, bad):
    images, texts, groups, splits = deepcopy(training_data)
    if bad == "group_leak":
        groups[8] = groups[0]
    elif bad == "missing_split":
        splits[8:] = ["train"] * 4
    elif bad == "nonfinite":
        images[0, 0] = np.nan
    elif bad == "zero_vector":
        images[0] = 0
    else:
        images = images[:, :20]
    with pytest.raises(AdapterError) as caught:
        train_adapter(images, texts, groups, splits, {"encoder_model": "cached-model"})
    assert not any(group in str(caught.value) for group in groups)


def test_pair_auc_assigns_half_credit_to_ties():
    assert _auc(np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_calibration_rejects_worse_full_head_and_preserves_cosine_ranking(training_data, monkeypatch):
    import semantic_image_stream.learned_relevance as module

    def overfit(*args):
        weights = np.zeros(FEATURE_DIM)
        weights[-1] = -8
        return weights, 0.0

    monkeypatch.setattr(module, "_fit", overfit)
    artifact, report = train_adapter(*training_data, {"encoder_model": "cached-encoder"})
    assert report["selection"]["chosen_head"] == "cosine_only"
    assert report["calibrated_score_only"] is True
    assert artifact["weights"][:-1] == [0.0] * 1024 and artifact["weights"][-1] > 0
    assert report["held_out"]["top1"] == report["held_out"]["baseline_top1"]
    assert report["held_out"]["pair_auc"] == report["held_out"]["baseline_pair_auc"]
    assert report["selection"]["candidates"]["full_pair"]["text_to_image_top1"] < 1
    assert report["selection"]["candidates"]["cosine_only"]["text_to_image_top1"] == 1


def test_json_roundtrip_and_runtime_inspection_do_not_load_or_write_models(tmp_path, trained):
    settings = Settings(training_dir=str(tmp_path), ranking="learned_context", relevance_threshold=0.40)
    path = tmp_path / "relevance-model.json"
    save_adapter(path, trained[0])
    before = path.read_bytes()
    model = load_adapter(path, settings.clip_model)
    vector = np.ones(512)
    scores = model.predict([vector, -vector], vector)
    assert scores.shape == (2,) and np.isfinite(scores).all()
    assert np.logical_and(scores >= 0, scores <= 1).all()
    info = inspect_training_pack(settings)
    assert info["status"] == "ready" and info["images"] == 12 and info["pairs"] == 36 and info["groups"] == 12
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]
    assert settings.relevance_threshold == 0.40


@pytest.mark.parametrize("change", [
    lambda data: data.update(version=True),
    lambda data: data.update(encoder_model="other-encoder"),
    lambda data: data.update(embedding_dim=256),
    lambda data: data.update(feature_checksum="wrong-feature-order"),
    lambda data: data.update(weights=[0] * 4),
    lambda data: data["weights"].__setitem__(0, float("nan")),
    lambda data: data["weights"].__setitem__(0, True),
    lambda data: data["calibration"].update(scale=-1),
    lambda data: data.update(recommended_threshold=2),
    lambda data: data["report"].update(train_ready=False),
    lambda data: data["report"]["held_out"].update(top1=float("inf")),
    lambda data: data["report"]["counts"]["test"].update(groups=1),
    lambda data: data["report"]["selection"].update(split="test"),
    lambda data: data["metadata"].update(encoder_fingerprint="not-a-fingerprint"),
])
def test_runtime_rejects_corrupt_incompatible_artifact(tmp_path, trained, change):
    data = deepcopy(trained[0])
    change(data)
    path = tmp_path / "relevance-model.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AdapterError, match="corrupt or incompatible"):
        load_adapter(path, "openai/clip-vit-base-patch32")


def test_missing_invalid_artifact_fails_before_encoder_load(tmp_path, monkeypatch):
    settings = Settings(training_dir=str(tmp_path), ranking="learned_context")

    def forbidden(*args, **kwargs):
        pytest.fail("Missing local model must never trigger an encoder load or network request")

    monkeypatch.setattr(ClipRanker, "_load", forbidden)
    ranker = build_ranker(settings)
    assert isinstance(ranker, LearnedContextRanker)
    with pytest.raises(AdapterError, match="No trained local relevance model"):
        ranker._load()
    assert inspect_training_pack(settings)["status"] == "missing"
    (tmp_path / "relevance-model.json").write_text("private broken caption", encoding="utf-8")
    info = inspect_training_pack(settings)
    assert info["status"] == "invalid" and "private" not in info["message"]


def test_local_only_encoder_cannot_start_download(tmp_path, monkeypatch):
    calls = []

    class NetworkBoundary:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(args)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        CLIPModel=NetworkBoundary, CLIPProcessor=NetworkBoundary))
    with pytest.raises(RuntimeError, match="not cached locally"):
        ClipRanker(Settings(cache_dir=str(tmp_path)))._load(local_only=True)
    assert calls == []


def cached_snapshot(settings):
    from pathlib import Path

    root = Path(settings.cache_dir) / "models" / "clip" / "models--openai--clip-vit-base-patch32" / "snapshots"
    snapshot = root / "complete"
    snapshot.mkdir(parents=True)
    for name in ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors"):
        (snapshot / name).write_text("fixture " + name, encoding="utf-8")
    return snapshot


@pytest.mark.parametrize("changed", ["model.safetensors", "config.json", "preprocessor_config.json",
                                     "tokenizer.json", "vocab.json", "merges.txt"])
def test_encoder_fingerprint_identifies_actual_weights_and_preprocessing(tmp_path, changed):
    settings = Settings(cache_dir=str(tmp_path))
    snapshot = cached_snapshot(settings)
    initial = encoder_fingerprint(settings)
    assert len(initial) == 64 and encoder_fingerprint(settings) == initial
    (snapshot / "pytorch_model.bin").write_text("unused older format")
    assert encoder_fingerprint(settings) == initial, "The unused weights must not identify the selected safetensors encoder"
    (snapshot / changed).write_text("changed cached bytes")
    assert encoder_fingerprint(settings) != initial


@pytest.mark.parametrize("problem", ["missing", "changed_weights", "changed_preprocessor", "valid"])
def test_learned_runtime_checks_exact_encoder_before_native_load(tmp_path, trained, monkeypatch, problem):
    settings = Settings(cache_dir=str(tmp_path / "cache"), training_dir=str(tmp_path / "training"))
    snapshot = cached_snapshot(settings)
    artifact = deepcopy(trained[0])
    artifact["metadata"]["encoder_fingerprint"] = encoder_fingerprint(settings)
    if problem == "missing":
        artifact["metadata"].pop("encoder_fingerprint")
    save_adapter(tmp_path / "training" / "relevance-model.json", artifact)
    if problem == "changed_weights":
        (snapshot / "model.safetensors").write_text("modified weights")
    elif problem == "changed_preprocessor":
        (snapshot / "preprocessor_config.json").write_text("different image preprocessing")
    calls = []
    monkeypatch.setattr(ClipRanker, "_load", lambda self, **kwargs: calls.append(kwargs))
    ranker = LearnedContextRanker(settings)
    if problem == "valid":
        ranker._load()
        assert calls == [{"local_only": True, "snapshot": snapshot}]
    else:
        with pytest.raises(AdapterError, match="encoder fingerprint|different CLIP weights or preprocessing"):
            ranker._load()
        assert calls == []
        if problem == "missing":
            assert inspect_training_pack(settings)["status"] == "invalid"


def test_learned_scores_include_scene_summary_and_full_history_with_existing_weights(tmp_path, trained):
    artifact = deepcopy(trained[0])
    artifact["weights"] = [0.0] * FEATURE_DIM
    artifact["weights"][-1], artifact["bias"] = 8.0, -4.0
    artifact["calibration"] = {"scale": 1.0, "intercept": 0.0}
    artifact["report"]["selection"]["chosen_head"] = "full_pair"
    artifact["report"]["calibrated_score_only"] = False
    save_adapter(tmp_path / "relevance-model.json", artifact)
    ranker = LearnedContextRanker(Settings(training_dir=str(tmp_path), relevance_threshold=0.40))

    class CPUVector:
        def __init__(self, values):
            self.values = np.asarray(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    scene, summary, history = np.eye(512)[:3]
    images = CPUVector([scene, summary])
    scores = ranker._score_vectors(images, [(CPUVector(scene), 0.5), (CPUVector(summary), 0.3), (CPUVector(history), 0.2)])
    matched, wrong = 1 / (1 + np.exp(-4)), 1 / (1 + np.exp(4))
    assert scores == pytest.approx([matched * 0.5 + wrong * 0.5, matched * 0.3 + wrong * 0.7])
    changed = ranker._score_vectors(images, [(CPUVector(scene), 0.5), (CPUVector(summary), 0.3), (CPUVector(summary), 0.2)])
    assert changed[1] > scores[1] and ranker.settings.relevance_threshold == 0.40
    ranker.processor = SimpleNamespace(tokenizer=SimpleNamespace(model_max_length=77,
        encode=lambda text, **kwargs: list(range(311))))
    assert [token for ids, _ in ranker._windows("complete long passage") for token in ids] == list(range(311))


def test_cosine_only_calibrates_after_context_average_to_preserve_scene_order(tmp_path, trained):
    artifact = deepcopy(trained[0])
    artifact["weights"] = [0.0] * FEATURE_DIM
    artifact["weights"][-1], artifact["bias"] = 20.0, -5.0
    artifact["calibration"] = {"scale": 1.0, "intercept": 0.0}
    artifact["report"]["selection"]["chosen_head"] = "cosine_only"
    artifact["report"]["calibrated_score_only"] = True
    save_adapter(tmp_path / "relevance-model.json", artifact)
    ranker = LearnedContextRanker(Settings(training_dir=str(tmp_path)))

    class CPUVector:
        def __init__(self, values):
            self.values = np.asarray(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    scene, background = np.eye(512)[:2]
    first, second = np.zeros(512), np.zeros(512)
    first[:3] = [0.44, 0.10, np.sqrt(1 - 0.44 ** 2 - 0.10 ** 2)]
    second[:3] = [0.26, 0.26, np.sqrt(1 - 2 * 0.26 ** 2)]
    scores = ranker._score_vectors(CPUVector([first, second]), [
        (CPUVector(scene), 0.5), (CPUVector(background), 0.3), (CPUVector(background), 0.2)])
    assert scores == pytest.approx([1 / (1 + np.exp(-0.4)), 1 / (1 + np.exp(-0.2))])
    assert scores[0] > scores[1], "Averaging sigmoid scores before contexts would incorrectly reverse these images"


def test_atomic_artifact_failure_keeps_previous_checked_model(tmp_path, trained, monkeypatch):
    import semantic_image_stream.learned_relevance as module

    path = tmp_path / "relevance-model.json"
    save_adapter(path, trained[0])
    old = path.read_bytes()

    def failed(*args):
        raise PermissionError("private path must stay hidden")

    monkeypatch.setattr(module.os, "replace", failed)
    with pytest.raises(AdapterError) as error:
        save_adapter(path, trained[0])
    assert "private" not in str(error.value)
    assert path.read_bytes() == old and not list(tmp_path.glob(".relevance-*.tmp"))
