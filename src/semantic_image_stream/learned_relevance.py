"""Small local relevance head on frozen CLIP features; no native model or pickle."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .training_paths import default_training_dir as default_training_dir
from .training_paths import training_file

EMBEDDING_DIM = 512
FEATURE_DIM = EMBEDDING_DIM * 2 + 1
FEATURE_CHECKSUM = hashlib.sha256(
    b"clip-pair-v1:l2-normalized-512:image*text,abs(image-text),cosine"
).hexdigest()
MAX_ARTIFACT_BYTES = 1_000_000
MAX_SAMPLES = 4000
SPLITS = ("train", "calibration", "test")


class AdapterError(RuntimeError):
    """Controlled diagnostics which never include captions or source paths."""


def _vectors(values):
    try:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM or not np.isfinite(matrix).all():
            raise ValueError
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 1e-8):
            raise ValueError
        return matrix / norms
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise AdapterError("Relevance training needs finite, nonzero 512-dimensional CLIP vectors.") from None


def pair_features(images, texts):
    image, text = _vectors(images), _vectors(texts)
    if len(image) != len(text):
        if len(image) == 1:
            image = np.broadcast_to(image, text.shape)
        elif len(text) == 1:
            text = np.broadcast_to(text, image.shape)
        else:
            raise AdapterError("Image and caption vectors must have matching lengths.")
    return np.concatenate((image * text, np.abs(image - text),
                           np.sum(image * text, axis=1, keepdims=True)), axis=1)


def _sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -60, 60)))


def _pairs(images, texts, groups, rng):
    similarities = images @ texts.T
    left, right, labels = list(range(len(images))), list(range(len(images))), [1] * len(images)
    groups = np.asarray(groups)
    for index in range(len(images)):
        eligible = np.flatnonzero(groups != groups[index])
        hard = int(eligible[np.argmax(similarities[index, eligible])])
        random_choices = eligible[eligible != hard]
        selected = [hard]
        if len(random_choices):
            selected.append(int(rng.choice(random_choices)))
        for negative in selected:
            left.append(index)
            right.append(negative)
            labels.append(0)
    left, right = np.asarray(left), np.asarray(right)
    features = pair_features(images[left], texts[right])
    return features, np.asarray(labels, dtype=np.float64), similarities[left, right]


def _fit(features, labels):
    # Standardization is fitted on training only and folded back into the
    # exported linear weights, keeping runtime inference one small dot product.
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 0.001)
    normalized = (features - mean) / scale
    weights = np.zeros(FEATURE_DIM)
    bias = math.log(labels.mean() / (1 - labels.mean()))
    first, second = np.zeros(FEATURE_DIM + 1), np.zeros(FEATURE_DIM + 1)
    for epoch in range(1, 181):
        residual = _sigmoid(normalized @ weights + bias) - labels
        gradient = np.append(normalized.T @ residual / len(labels) + 0.02 * weights, residual.mean())
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient ** 2
        step = 0.03 * (first / (1 - 0.9 ** epoch)) / (np.sqrt(second / (1 - 0.999 ** epoch)) + 1e-8)
        weights -= step[:-1]
        bias -= step[-1]
    folded = weights / scale
    return folded, float(bias - mean @ folded)


def _calibrate(logits, labels):
    inputs = np.column_stack((logits, np.ones(len(logits))))
    parameters = np.array([1.0, 0.0])
    anchor = parameters.copy()
    for _ in range(60):
        probability = _sigmoid(inputs @ parameters)
        gradient = inputs.T @ (probability - labels) / len(labels) + 0.002 * (parameters - anchor)
        curvature = probability * (1 - probability)
        hessian = (inputs.T * curvature) @ inputs / len(labels) + np.eye(2) * 0.002
        step = np.clip(np.linalg.solve(hessian, gradient), -1, 1)
        parameters -= step
        parameters[0] = np.clip(parameters[0], 0.02, 20)
        parameters[1] = np.clip(parameters[1], -60, 60)
        if np.max(np.abs(step)) < 1e-7:
            break
    return {"scale": float(parameters[0]), "intercept": float(parameters[1])}


def _auc(labels, scores):
    # Average ranks give ties half credit; no scipy/sklearn training dependency.
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    begin = 0
    while begin < len(scores):
        end = begin + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[begin]:
            end += 1
        ranks[order[begin:end]] = (begin + 1 + end) / 2
        begin = end
    positive = labels == 1
    count = int(positive.sum())
    return float((ranks[positive].sum() - count * (count + 1) / 2) / (count * (len(labels) - count)))


def _threshold(labels, scores):
    best = None
    for threshold in np.unique(np.r_[0.0, scores, 1.0]):
        predicted = scores >= threshold
        true = int(np.sum(predicted & (labels == 1)))
        precision = true / max(1, int(predicted.sum()))
        recall = true / int(np.sum(labels == 1))
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        candidate = (f1, precision, recall, float(threshold))
        if best is None or candidate > best:
            best = candidate
    return best[3], {"f1": best[0], "precision": best[1], "recall": best[2]}


class Adapter:
    def __init__(self, artifact):
        self.artifact = artifact
        self.weights = np.asarray(artifact["weights"], dtype=np.float64)
        self.bias = artifact["bias"]
        self.calibration = artifact["calibration"]

    def predict(self, images, texts):
        logits = pair_features(images, texts) @ self.weights + self.bias
        return _sigmoid(logits * self.calibration["scale"] + self.calibration["intercept"])

    def predict_cosines(self, cosines):
        """Apply the monotonic trained scalar after full-context aggregation."""
        if self.artifact["report"]["selection"]["chosen_head"] != "cosine_only":
            raise AdapterError("Weighted cosine calibration requires a cosine-only relevance head.")
        values = np.asarray(cosines, dtype=np.float64)
        if not np.isfinite(values).all():
            raise AdapterError("Local relevance features must be finite.")
        logits = values * self.weights[-1] + self.bias
        return _sigmoid(logits * self.calibration["scale"] + self.calibration["intercept"])


def _top1(adapter, images, texts, groups, direction="text_to_image"):
    correct, baseline = 0, 0
    queries = texts if direction == "text_to_image" else images
    for index, query in enumerate(queries):
        scores = (adapter.predict(images, query) if direction == "text_to_image" else
                  adapter.predict(query, texts))
        correct += groups[int(np.argmax(scores))] == groups[index]
        baseline_scores = images @ query if direction == "text_to_image" else texts @ query
        baseline += groups[int(np.argmax(baseline_scores))] == groups[index]
    return float(correct / len(images)), float(baseline / len(images))


def train_adapter(image_vectors, text_vectors, group_ids, split_labels, metadata):
    """Train on train groups, calibrate separately, and report untouched test pairs.

    Caption pairing supplies weak labels: nearest wrong-source captions can
    still describe an image. Curated story evaluation is needed to assess actual
    audiobook relevance; this report never asserts improvement automatically.
    """
    images, texts = _vectors(image_vectors), _vectors(text_vectors)
    groups, splits = list(group_ids), list(split_labels)
    if (not 6 <= len(images) <= MAX_SAMPLES or len(texts) != len(images)
            or len(groups) != len(images) or len(splits) != len(images)
            or any(not isinstance(group, str) or not group or len(group) > 512 for group in groups)
            or any(split not in SPLITS for split in splits)):
        raise AdapterError("Provide bounded paired vectors and explicit train/calibration/test source groups.")
    ownership = {}
    for group, split in zip(groups, splits):
        if group in ownership and ownership[group] != split:
            raise AdapterError("A source-image group cannot appear in more than one training split.")
        ownership[group] = split
    if not isinstance(metadata, dict) or not isinstance(metadata.get("encoder_model"), str) or not metadata["encoder_model"]:
        raise AdapterError("Record the exact frozen CLIP encoder model used for training.")
    try:
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError, OverflowError):
        raise AdapterError("Training provenance must be finite JSON data.") from None
    rng = np.random.default_rng(20261013)
    batches, counts = {}, {}
    for split in SPLITS:
        indices = [index for index, value in enumerate(splits) if value == split]
        selected_groups = [groups[index] for index in indices]
        if len(set(selected_groups)) < 2:
            raise AdapterError("Each split needs at least two distinct source-image groups.")
        batch = _pairs(images[indices], texts[indices], selected_groups, rng)
        batches[split] = (*batch, indices, selected_groups)
        counts[split] = {"images": len(indices), "groups": len(set(selected_groups)), "pairs": len(batch[1]),
                         "positives": len(indices), "negatives": len(batch[1]) - len(indices)}
    train_features, train_labels = batches["train"][:2]
    full_weights, full_bias = _fit(train_features, train_labels)
    # A high-dimensional head can overfit a small public subset. Fit a second
    # monotonic two-parameter head on training cosine only, preserving original
    # retrieval ordering when calibration cannot support a stronger pair head.
    cosine_fit = _calibrate(train_features[:, -1], train_labels)
    cosine_weights = np.zeros(FEATURE_DIM)
    cosine_weights[-1] = cosine_fit["scale"]
    candidate_heads = {"full_pair": (full_weights, full_bias),
                       "cosine_only": (cosine_weights, cosine_fit["intercept"])}
    candidates, evidence = {}, {}
    cal_features, cal_labels, _, cal_indices, cal_groups = batches["calibration"]
    for method, (weights, bias) in candidate_heads.items():
        calibration = _calibrate(cal_features @ weights + bias, cal_labels)
        temporary = {"weights": weights.tolist(), "bias": bias, "calibration": calibration}
        model = Adapter(temporary)
        top1, _ = _top1(model, images[cal_indices], texts[cal_indices], cal_groups)
        inverse, _ = _top1(model, images[cal_indices], texts[cal_indices], cal_groups, "image_to_text")
        scores = _sigmoid((cal_features @ weights + bias) * calibration["scale"] + calibration["intercept"])
        evidence[method] = {"text_to_image_top1": top1, "image_to_text_top1": inverse,
                            "pair_auc": _auc(cal_labels, scores)}
        candidates[method] = (weights, bias, calibration, scores)
    chosen = "full_pair" if (evidence["full_pair"]["text_to_image_top1"] > evidence["cosine_only"]["text_to_image_top1"]
                              and evidence["full_pair"]["pair_auc"] > evidence["cosine_only"]["pair_auc"]) else "cosine_only"
    weights, bias, calibration, calibrated = candidates[chosen]
    artifact = {"version": 1, "encoder_model": metadata["encoder_model"], "embedding_dim": EMBEDDING_DIM,
                "feature_checksum": FEATURE_CHECKSUM, "weights": weights.tolist(), "bias": bias,
                "calibration": calibration, "metadata": metadata}
    adapter = Adapter(artifact)
    threshold, operating = _threshold(cal_labels, calibrated)
    features, labels, cosines, indices, selected_groups = batches["test"]
    learned = _sigmoid((features @ weights + bias) * calibration["scale"] + calibration["intercept"])
    top1, baseline = _top1(adapter, images[indices], texts[indices], selected_groups)
    inverse, inverse_baseline = _top1(adapter, images[indices], texts[indices], selected_groups, "image_to_text")
    report = {"train_ready": True, "seed": 20261013, "epochs": 180, "l2": 0.02,
              "label_method": "paired-caption positives; wrong-source random and cosine-hard negatives",
              "counts": counts, "threshold_method": "calibration F1",
              "top1_direction": "text_to_image", "calibrated_score_only": chosen == "cosine_only",
              "selection": {"split": "calibration", "chosen_head": chosen,
                            "criterion": "Full pair head must improve calibration text-to-image top1 and pair AUC; otherwise cosine-only",
                            "candidates": evidence},
              "calibration": {"threshold": threshold, **operating},
              "held_out": {"top1": top1, "baseline_top1": baseline, "pair_auc": _auc(labels, learned),
                           "baseline_pair_auc": _auc(labels, cosines),
                           "image_to_text_top1": inverse, "baseline_image_to_text_top1": inverse_baseline,
                           "brier": float(np.mean((learned - labels) ** 2))}}
    artifact["recommended_threshold"], artifact["report"] = threshold, report
    _checked(artifact, metadata["encoder_model"])
    return artifact, report


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _checked(artifact, encoder_model):
    try:
        if (not isinstance(artifact, dict) or type(artifact.get("version")) is not int or artifact["version"] != 1
                or not isinstance(artifact.get("encoder_model"), str) or not 0 < len(artifact["encoder_model"]) <= 200
                or artifact.get("encoder_model") != encoder_model or artifact.get("embedding_dim") != EMBEDDING_DIM
                or artifact.get("feature_checksum") != FEATURE_CHECKSUM):
            raise ValueError
        weights = artifact["weights"]
        calibration = artifact["calibration"]
        report = artifact["report"]
        selection = report["selection"]
        metadata = artifact.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError
        if "encoder_fingerprint" in metadata and (not isinstance(metadata["encoder_fingerprint"], str)
                or not re.fullmatch(r"[a-f0-9]{64}", metadata["encoder_fingerprint"])):
            raise ValueError
        if (not isinstance(weights, list) or len(weights) != FEATURE_DIM or not all(_number(weight) for weight in weights)
                or not _number(artifact["bias"]) or not isinstance(calibration, dict)
                or not _number(calibration["scale"]) or not 0 < calibration["scale"] <= 100
                or not _number(calibration["intercept"]) or not _number(artifact["recommended_threshold"])
                or not 0 <= artifact["recommended_threshold"] <= 1 or report.get("train_ready") is not True):
            raise ValueError
        if (selection.get("split") != "calibration" or selection.get("chosen_head") not in {"full_pair", "cosine_only"}
                or report.get("top1_direction") != "text_to_image"
                or report.get("calibrated_score_only") is not (selection["chosen_head"] == "cosine_only")):
            raise ValueError
        if selection["chosen_head"] == "cosine_only" and (any(weights[:-1]) or weights[-1] <= 0):
            raise ValueError
        for split in SPLITS:
            counts = report["counts"][split]
            if any(type(counts[key]) is not int or counts[key] < 1
                   for key in ("images", "groups", "pairs", "positives", "negatives")):
                raise ValueError
            if (not 2 <= counts["groups"] <= counts["images"] <= MAX_SAMPLES
                    or counts["positives"] != counts["images"]
                    or counts["pairs"] != counts["positives"] + counts["negatives"]
                    or counts["negatives"] > counts["images"] * 2):
                raise ValueError
        if any(not _number(report["held_out"][key]) or not 0 <= report["held_out"][key] <= 1
               for key in ("top1", "baseline_top1", "pair_auc", "baseline_pair_auc", "brier")):
            raise ValueError
        if len(json.dumps(artifact, allow_nan=False).encode("utf-8")) > MAX_ARTIFACT_BYTES:
            raise ValueError
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        raise AdapterError("The local relevance model is corrupt or incompatible with the selected CLIP encoder. Train it again.") from None
    return Adapter(artifact)


def load_adapter(path, encoder_model):
    try:
        with Path(path).open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            if size > MAX_ARTIFACT_BYTES:
                raise ValueError("Relevance model exceeds the size bound")
            raw = source.read(size + 1)
            if len(raw) > size or source.read(1):
                raise ValueError("Relevance model changed while reading")
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ValueError
        artifact = json.loads(raw)
    except FileNotFoundError:
        raise AdapterError("No trained local relevance model was found. Train the local model first.") from None
    except (OSError, ValueError, TypeError, RecursionError, MemoryError):
        raise AdapterError("The local relevance model could not be read. Train it again.") from None
    return _checked(artifact, encoder_model)


def save_adapter(path, artifact):
    _checked(artifact, artifact.get("encoder_model"))
    target = Path(path)
    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".relevance-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(artifact, output, ensure_ascii=False, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    except (OSError, ValueError, TypeError):
        raise AdapterError("The trained relevance model could not be saved locally.") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def inspect_training_pack(settings):
    path = training_file(settings, "relevance-model.json")
    try:
        adapter = load_adapter(path, settings.clip_model)
        if not adapter.artifact.get("metadata", {}).get("encoder_fingerprint"):
            raise AdapterError("The local relevance model has no exact encoder fingerprint. Train it again with the cached encoder.")
    except AdapterError as error:
        return {"status": "invalid" if path.exists() else "missing", "message": str(error), "images": 0}
    artifact = adapter.artifact
    count = sum(values["images"] for values in artifact["report"]["counts"].values())
    return {"status": "ready", "model": artifact["encoder_model"], "images": count,
            "pairs": sum(values["pairs"] for values in artifact["report"]["counts"].values()),
            "groups": sum(values["groups"] for values in artifact["report"]["counts"].values()),
            "chosen_head": artifact["report"]["selection"]["chosen_head"],
            "recommended_threshold": artifact["recommended_threshold"],
            "message": f"Local relevance model ready · {count} image-caption examples."}
