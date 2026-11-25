"""Simulated 8-channel electronic nose (see ALGORITHMS.md §2)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
BANK_PATH = MODELS_DIR / "sensor_bank.json"
SAMPLES_PATH = ROOT / "data" / "samples.json"
DATASET_PATH = ROOT / "data" / "curated_GS_LF_merged_4983.csv"

N_SENSORS = 8
SENSOR_NAMES = [f"S{i}" for i in range(1, N_SENSORS + 1)]
FEATURE_NAMES = [
    "mw",
    "logp",
    "o_frac",
    "s_count",
    "n_count",
    "aromatic_rings",
    "tpsa",
    "hbd",
]
DEFAULT_SEED = 7


@dataclass
class SensorBank:
    seed: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    gain: np.ndarray
    humidity_coeff: np.ndarray
    drift_coeff: np.ndarray

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "n_sensors": N_SENSORS,
            "feature_names": FEATURE_NAMES,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "gain": self.gain.tolist(),
            "humidity_coeff": self.humidity_coeff.tolist(),
            "drift_coeff": self.drift_coeff.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SensorBank":
        return cls(
            seed=int(data["seed"]),
            feature_mean=np.asarray(data["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(data["feature_std"], dtype=np.float64),
            weights=np.asarray(data["weights"], dtype=np.float64),
            bias=np.asarray(data["bias"], dtype=np.float64),
            gain=np.asarray(data["gain"], dtype=np.float64),
            humidity_coeff=np.asarray(data["humidity_coeff"], dtype=np.float64),
            drift_coeff=np.asarray(data["drift_coeff"], dtype=np.float64),
        )


@lru_cache(maxsize=8192)
def _molecule_features(smiles: str) -> tuple[float, ...]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")
    heavy = mol.GetNumHeavyAtoms()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return (
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        symbols.count("O") / heavy if heavy else 0.0,
        float(symbols.count("S")),
        float(symbols.count("N")),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(rdMolDescriptors.CalcNumHBD(mol)),
    )


def molecule_features(smiles: str) -> np.ndarray:
    """Property vector for one molecule; raises ValueError on invalid SMILES."""
    return np.asarray(_molecule_features(smiles), dtype=np.float64)


def _dataset_smiles() -> list[str]:
    if not DATASET_PATH.exists():
        return []
    with DATASET_PATH.open(newline="", encoding="utf-8") as handle:
        return [
            row["nonStereoSMILES"]
            for row in csv.DictReader(handle)
            if row.get("nonStereoSMILES")
        ]


def _sample_smiles() -> list[str]:
    data = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    return [m["smiles"] for sample in data["samples"] for m in sample["molecules"]]


def _feature_stats() -> tuple[np.ndarray, np.ndarray]:
    """Standardization stats from dataset + sample molecules (stable at runtime)."""
    vectors: list[np.ndarray] = []
    for smiles in _sample_smiles() + _dataset_smiles():
        try:
            vectors.append(molecule_features(smiles))
        except ValueError:
            continue
    features = np.vstack(vectors)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def make_sensor_bank(seed: int = DEFAULT_SEED) -> SensorBank:
    """Fixed seeded sensor bank; redraws until no weight pair exceeds 0.9 cosine."""
    mean, std = _feature_stats()
    final_seed = seed
    while True:
        rng = np.random.default_rng(final_seed)
        weights = rng.normal(0.0, 1.0, (N_SENSORS, len(FEATURE_NAMES)))
        for i in range(N_SENSORS):
            # Real metal-oxide sensors key on a few properties, not all of them.
            weights[i, rng.choice(len(FEATURE_NAMES), size=4, replace=False)] = 0.0
        norms = np.linalg.norm(weights, axis=1)
        cosine = (weights @ weights.T) / np.outer(norms, norms)
        if np.max(cosine - np.eye(N_SENSORS)) <= 0.9:
            break
        final_seed += 1

    return SensorBank(
        seed=final_seed,
        feature_mean=mean,
        feature_std=std,
        weights=weights,
        bias=rng.normal(0.0, 0.5, N_SENSORS),
        gain=rng.uniform(0.4, 1.0, N_SENSORS),
        humidity_coeff=rng.uniform(0.05, 0.25, N_SENSORS),
        drift_coeff=rng.uniform(0.00, 0.03, N_SENSORS),
    )


def save_bank(bank: SensorBank, path: Path = BANK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank.to_dict(), indent=2), encoding="utf-8")


def load_bank() -> SensorBank:
    """Load the committed bank, or generate and persist it on first use."""
    if BANK_PATH.exists():
        return SensorBank.from_dict(json.loads(BANK_PATH.read_text(encoding="utf-8")))
    bank = make_sensor_bank()
    save_bank(bank)
    return bank


_BANK: SensorBank | None = None


def _bank() -> SensorBank:
    global _BANK
    if _BANK is None:
        _BANK = load_bank()
    return _BANK


def simulate(
    molecules: list[dict],
    noise: float,
    humidity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return 8 readings relative to clean air (1.0 means no change)."""
    bank = _bank()
    if not molecules:
        raise ValueError("simulate() needs at least one molecule")

    raw_weights = np.asarray([float(m["weight"]) for m in molecules], dtype=np.float64)
    weights = raw_weights / raw_weights.sum()
    features = np.vstack([molecule_features(m["smiles"]) for m in molecules])
    z = (features - bank.feature_mean) / bank.feature_std

    volatility = np.clip(np.exp(-(features[:, 0] - 100.0) / 120.0), 0.2, 1.5)
    affinity = 1.0 / (1.0 + np.exp(-(z @ bank.weights.T + bank.bias)))
    intensity = rng.uniform(0.6, 1.4)
    signal = ((weights * volatility) @ affinity) * bank.gain * intensity

    saturation = 1.0 - np.exp(-3.0 * signal)
    environment = bank.humidity_coeff * (humidity - 0.5)
    drift = bank.drift_coeff * rng.uniform(-1.0, 1.0, N_SENSORS)
    sensor_noise = rng.normal(0.0, noise, N_SENSORS)
    return np.clip(1.0 + saturation + environment + drift + sensor_noise, 0.5, None)


if __name__ == "__main__":
    bank = load_bank()
    print(
        f"sensor bank: seed={bank.seed} shape={bank.weights.shape} "
        f"-> {BANK_PATH.relative_to(ROOT)}"
    )
