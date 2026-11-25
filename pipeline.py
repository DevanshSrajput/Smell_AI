"""Join the simulated nose with the molecule smell model (ALGORITHMS.md §5)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

import sensor
import smell_model

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
SAMPLES_PATH = ROOT / "samples.json"
LOG_PATH = ROOT / "logs" / "sniffs.jsonl"

REQUIRED_FILES = {
    "models/smell.joblib": "python smell_model.py",
    "models/descriptors.json": "python smell_model.py",
    "models/smell_metrics.json": "python smell_model.py",
    "models/sensor_bank.json": "python train_nose.py",
    "models/nose.joblib": "python train_nose.py",
    "models/nose_metrics.json": "python train_nose.py",
}


@dataclass
class Sample:
    id: str
    label: str
    molecules: list[dict]
    weights: np.ndarray


@dataclass
class Context:
    samples: dict[str, Sample]
    bank: sensor.SensorBank
    nose: Any
    nose_classes: list[str]
    smell_model: Any
    descriptors: list[str]
    base_rate: np.ndarray
    mol_probs: dict[str, np.ndarray]


def missing_model_files() -> list[str]:
    return [rel for rel in REQUIRED_FILES if not (ROOT / rel).exists()]


def require_models() -> None:
    missing = missing_model_files()
    if not missing:
        return
    lines = "\n".join(f"  {rel}  (run: {REQUIRED_FILES[rel]})" for rel in missing)
    raise FileNotFoundError(
        f"Missing model files:\n{lines}\n\nTrain the models first:\n"
        "  python smell_model.py\n  python train_nose.py"
    )


def load_samples(path: Path = SAMPLES_PATH) -> dict[str, Sample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: dict[str, Sample] = {}
    for raw in data["samples"]:
        sample_id = raw["id"]
        if sample_id in samples:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        molecules = raw["molecules"]
        if not molecules:
            raise ValueError(f"Sample {sample_id} has no molecules")
        for molecule in molecules:
            try:
                sensor.molecule_features(molecule["smiles"])
            except ValueError as exc:
                raise ValueError(
                    f"Sample {sample_id}: could not parse {molecule['name']} "
                    f"({molecule['smiles']!r})"
                ) from exc
        weights = np.asarray([float(m["weight"]) for m in molecules], dtype=np.float64)
        samples[sample_id] = Sample(
            id=sample_id,
            label=raw["label"],
            molecules=molecules,
            weights=weights / weights.sum(),
        )
    return samples


def load_all() -> Context:
    require_models()
    samples = load_samples()
    bank = sensor.load_bank()
    nose = joblib.load(MODELS_DIR / "nose.joblib")
    nose_classes = [str(c) for c in nose.classes_]
    if set(nose_classes) != set(samples):
        raise ValueError(
            "Nose classes do not match samples.json; rerun: python train_nose.py"
        )
    smell, descriptors = smell_model.load_model()
    descriptor_data = json.loads(
        (MODELS_DIR / "descriptors.json").read_text(encoding="utf-8")
    )
    base_rate = np.asarray(
        [descriptor_data["base_rate"][name] for name in descriptors], dtype=np.float64
    )

    mol_probs = {
        sample_id: smell_model.predict_proba(
            smell, [m["smiles"] for m in sample.molecules]
        )
        for sample_id, sample in samples.items()
    }
    return Context(
        samples=samples,
        bank=bank,
        nose=nose,
        nose_classes=nose_classes,
        smell_model=smell,
        descriptors=descriptors,
        base_rate=base_rate,
        mol_probs=mol_probs,
    )


def describe(ctx: Context, sample_id: str, top_k: int = 5) -> list[dict]:
    """Rank descriptors by score * sqrt(lift) over the sample's molecule mix."""
    sample = ctx.samples[sample_id]
    scores = sample.weights @ ctx.mol_probs[sample_id]
    lift = scores / np.maximum(ctx.base_rate, 0.005)
    order = np.argsort(-(scores * np.sqrt(lift)))[:top_k]
    return [
        {
            "descriptor": ctx.descriptors[i],
            "score": float(scores[i]),
            "lift": float(lift[i]),
        }
        for i in order
    ]


def molecule_descriptors(ctx: Context, smiles: str, top_k: int = 8) -> list[dict]:
    """Top descriptors for one molecule; raises ValueError on invalid SMILES."""
    if not smiles.strip():
        raise ValueError("Could not parse that SMILES.")
    try:
        scores = smell_model.predict_proba(ctx.smell_model, [smiles])[0]
    except ValueError as exc:
        raise ValueError("Could not parse that SMILES.") from exc
    lift = scores / np.maximum(ctx.base_rate, 0.005)
    order = np.argsort(-(scores * np.sqrt(lift)))[:top_k]
    return [
        {
            "descriptor": ctx.descriptors[i],
            "score": float(scores[i]),
            "lift": float(lift[i]),
        }
        for i in order
    ]


def _log(result: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **result}
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def sniff(
    ctx: Context,
    sample_name: str,
    noise: float = 0.03,
    humidity: float = 0.5,
    seed: int | None = None,
) -> dict:
    if sample_name not in ctx.samples:
        raise KeyError(f"Unknown sample {sample_name!r}; choose from {list(ctx.samples)}")
    rng = np.random.default_rng(seed)
    sample = ctx.samples[sample_name]
    readings = sensor.simulate(sample.molecules, noise, humidity, rng)

    probabilities = ctx.nose.predict_proba(readings.reshape(1, -1))[0]
    predicted = ctx.nose_classes[int(np.argmax(probabilities))]
    result = {
        "presented": sample_name,
        "conditions": {"noise": float(noise), "humidity": float(humidity), "seed": seed},
        "sensors": {
            "names": list(sensor.SENSOR_NAMES),
            "values": [float(value) for value in readings],
        },
        "nose": {
            "predicted": predicted,
            "confidence": float(np.max(probabilities)),
            "probabilities": {
                name: float(probability)
                for name, probability in zip(ctx.nose_classes, probabilities)
            },
            "correct": predicted == sample_name,
        },
        "description": describe(ctx, predicted, top_k=5),
        "molecules": ctx.samples[predicted].molecules,
    }
    _log(result)
    return result


if __name__ == "__main__":
    context = load_all()
    print(f"loaded {len(context.samples)} samples, {len(context.descriptors)} descriptors")
    for sample_id in context.samples:
        result = sniff(context, sample_id, seed=1)
        top = ", ".join(item["descriptor"] for item in result["description"])
        print(f"{sample_id:15s} -> {result['nose']['predicted']:15s} {top}")
