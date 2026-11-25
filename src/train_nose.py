"""Train the simulated-nose classifier on generated sensor sessions."""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import sensor

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SAMPLES_PATH = ROOT / "data" / "samples.json"
N_SESSIONS = 200
NOISE_LEVELS = (0.03, 0.10, 0.20, 0.30)
SEED = 1


def _samples() -> list[dict]:
    data = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    return data["samples"]


def _collect(
    samples: list[dict],
    n_sessions: int,
    rng: np.random.Generator,
    sigma_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[str] = []
    groups: list[int] = []
    for sample in samples:
        for session in range(n_sessions):
            sigma = rng.uniform(*sigma_range)
            humidity = rng.uniform(0.2, 0.8)
            X.append(sensor.simulate(sample["molecules"], sigma, humidity, rng))
            y.append(sample["id"])
            groups.append(session)
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def build_dataset(
    n_sessions: int = N_SESSIONS, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sessions: noise sigma ~ U(0.01, 0.06), humidity ~ U(0.2, 0.8)."""
    rng = np.random.default_rng(seed)
    return _collect(_samples(), n_sessions, rng, (0.01, 0.06))


def _accuracy_at_noise(
    model: RandomForestClassifier, samples: list[dict], sigma: float, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    X, y, _ = _collect(samples, 300, rng, (sigma, sigma))
    return float(accuracy_score(y, model.predict(X)))


def train(seed: int = SEED) -> dict:
    t_start = time.perf_counter()
    samples = _samples()
    classes = [sample["id"] for sample in samples]
    bank = sensor.load_bank()
    X, y, _groups = build_dataset(seed=seed)

    # Every row is one session; stratifying keeps all six samples represented.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    matrix = confusion_matrix(y_test, predictions, labels=classes)
    coffee_index, wood_index = classes.index("coffee"), classes.index("wood_smoke")

    curve = {
        f"{sigma:.2f}": _accuracy_at_noise(model, samples, sigma, seed + 100 + i)
        for i, sigma in enumerate(NOISE_LEVELS)
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "nose.joblib", compress=3)
    metrics = {
        "n_sessions_per_sample": N_SESSIONS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model": {"type": "RandomForestClassifier", "n_estimators": 200},
        "accuracy_default_noise": accuracy,
        "accuracy_by_noise": curve,
        "classes": classes,
        "confusion_matrix": matrix.tolist(),
        "coffee_wood_smoke_confusion": {
            "coffee_as_wood_smoke": int(matrix[coffee_index, wood_index]),
            "wood_smoke_as_coffee": int(matrix[wood_index, coffee_index]),
        },
        "seeds": {"sensor_bank": bank.seed, "data": seed, "model": seed},
        "train_seconds": round(time.perf_counter() - t_start, 2),
    }
    (MODELS_DIR / "nose_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"held-out accuracy: {accuracy:.3f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(matrix)
    print("accuracy by noise sigma:", {k: round(v, 3) for k, v in curve.items()})
    print("saved models/nose.joblib")
    return metrics


if __name__ == "__main__":
    train()
