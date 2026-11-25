"""Train and serve the OpenPOM 138-descriptor RandomForest smell model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent.parent
# Approved URL (TRD.md §3.1 omitted /curated_datasets/ and returns 404).
DATA_URL = (
    "https://raw.githubusercontent.com/BioMachineLearning/openpom/"
    "main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
)
DEFAULT_CSV = ROOT / "data" / "curated_GS_LF_merged_4983.csv"
MODELS_DIR = ROOT / "models"
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _download_csv(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(DATA_URL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(
            f"Download failed ({exc}).\n"
            f"URL: {DATA_URL}\n"
            f"Target: {dest}\n"
            f"Provide a local file via: python smell_model.py --csv PATH",
            file=sys.stderr,
        )
        sys.exit(1)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)


def load_dataset(csv_path: str | None = None) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (smiles, Y[n, 138], descriptor_names). Downloads to data/ if missing."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV
    if csv_path is None and not path.exists():
        _download_csv(path)

    df = pd.read_csv(path)
    if "nonStereoSMILES" not in df.columns:
        raise ValueError(f"Missing 'nonStereoSMILES' column; found: {list(df.columns)}")

    label_cols = [
        col
        for col in df.columns
        if col != "nonStereoSMILES"
        and pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
        and df[col].notna().all()
        and len(df[col]) > 0
        and set(df[col].unique()).issubset({0, 1})
    ]

    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"Row count: {len(df)}")
    print(f"Detected binary label columns: {len(label_cols)}")
    assert len(label_cols) == 138, (
        f"Expected 138 binary label columns, found {len(label_cols)}: {label_cols}"
    )

    smiles = df["nonStereoSMILES"].tolist()
    Y = df[label_cols].to_numpy(dtype=np.int64)
    return smiles, Y, label_cols


def featurize(smiles: list[str]) -> tuple[np.ndarray, list[int]]:
    """Morgan count fingerprints (radius=2, 2048 bits). Drops unparsable SMILES."""
    fps: list[np.ndarray] = []
    kept: list[int] = []
    for idx, sm in enumerate(smiles):
        if not isinstance(sm, str) or not sm.strip():
            continue
        mol = Chem.MolFromSmiles(sm)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        fps.append(MORGAN_GEN.GetCountFingerprintAsNumPy(mol))
        kept.append(idx)

    # ponytail: raw float32 counts (no log1p); RF is scale-invariant on counts
    X = np.asarray(fps, dtype=np.float32) if fps else np.empty((0, 2048), dtype=np.float32)
    return X, kept


def _proba_matrix(model: Any, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    if isinstance(raw, np.ndarray):
        return raw.astype(np.float64, copy=False)
    cols = [
        p[:, 1]
        if p.shape[1] == 2
        else (np.ones(len(X)) if model.classes_[i][0] == 1 else np.zeros(len(X)))
        for i, p in enumerate(raw)
    ]
    return np.column_stack(cols)


def _eval_auc(
    Y_true: np.ndarray, P_pred: np.ndarray, label_cols: list[str]
) -> tuple[float, dict[str, float]]:
    per_auc: dict[str, float] = {}
    for j, name in enumerate(label_cols):
        if len(np.unique(Y_true[:, j])) == 2:
            per_auc[name] = float(roc_auc_score(Y_true[:, j], P_pred[:, j]))
    macro = float(np.mean(list(per_auc.values()))) if per_auc else 0.0
    return macro, per_auc


def train(csv_path: str | None = None, seed: int = 42) -> dict:
    """Train, evaluate on 20% held-out split, save model & metrics."""
    t_start = time.perf_counter()
    smiles, Y, label_cols = load_dataset(csv_path)
    n_total = len(smiles)
    X, kept = featurize(smiles)
    Y = Y[kept]
    n_used = len(kept)
    n_dropped = n_total - n_used
    print(f"Dropped unparsable SMILES: {n_dropped}")

    # ponytail: random 80/20 split per ALGORITHMS.md §1.3; switch to scaffold split for novel-scaffold eval
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )

    rf_params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
    }
    t_fit = time.perf_counter()
    clf: Any = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, Y_train)
    fit_seconds = time.perf_counter() - t_fit

    # Time budget fallback per ALGORITHMS.md §1.2 / TRD.md §5
    if fit_seconds > 120 or (time.perf_counter() - t_start) > 180:
        rf_params["n_estimators"] = 150
        t_fit = time.perf_counter()
        clf = RandomForestClassifier(
            n_estimators=150,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        )
        clf.fit(X_train, Y_train)
        fit_seconds = time.perf_counter() - t_fit
        if fit_seconds > 120:
            rf_params["max_depth"] = 30
            t_fit = time.perf_counter()
            clf = RandomForestClassifier(
                n_estimators=150,
                max_depth=30,
                n_jobs=-1,
                random_state=seed,
                class_weight="balanced_subsample",
            )
            clf.fit(X_train, Y_train)
            fit_seconds = time.perf_counter() - t_fit

    model_info: dict[str, Any] = {"type": "RandomForestClassifier", **rf_params}
    P_test = _proba_matrix(clf, X_test)
    macro_auc, per_auc = _eval_auc(Y_test, P_test, label_cols)

    # Underperformance fallback per ALGORITHMS.md §1.2 (carved from train, never test)
    if macro_auc < 0.75:
        X_tr_sub, X_val, Y_tr_sub, Y_val = train_test_split(
            X_train, Y_train, test_size=0.2, random_state=seed
        )
        rf_val = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        ).fit(X_tr_sub, Y_tr_sub)
        rf_val_auc, _ = _eval_auc(Y_val, _proba_matrix(rf_val, X_val), label_cols)

        mlp_val = MLPClassifier(
            hidden_layer_sizes=(512,), early_stopping=True, random_state=seed
        ).fit(X_tr_sub, Y_tr_sub)
        mlp_val_auc, _ = _eval_auc(Y_val, _proba_matrix(mlp_val, X_val), label_cols)

        if mlp_val_auc > rf_val_auc:
            t_fit = time.perf_counter()
            clf = MLPClassifier(
                hidden_layer_sizes=(512,), early_stopping=True, random_state=seed
            ).fit(X_train, Y_train)
            fit_seconds = time.perf_counter() - t_fit
            model_info = {"type": "MLPClassifier", "hidden_layer_sizes": [512], "early_stopping": True}
            P_test = _proba_matrix(clf, X_test)
            macro_auc, per_auc = _eval_auc(Y_test, P_test, label_cols)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # ponytail: compress=3 shrinks multi-output RF >10x on disk with transparent load
    joblib.dump(clf, MODELS_DIR / "smell.joblib", compress=3)

    base_rate = {
        name: float(rate) for name, rate in zip(label_cols, Y_train.mean(axis=0))
    }
    (MODELS_DIR / "descriptors.json").write_text(
        json.dumps({"names": label_cols, "base_rate": base_rate}, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "n_molecules_total": int(n_total),
        "n_molecules_used": int(n_used),
        "n_dropped_unparsable": int(n_dropped),
        "n_descriptors": len(label_cols),
        "featurizer": {"type": "morgan_count", "radius": 2, "n_bits": 2048},
        "model": model_info,
        "split": {"type": "random", "test_fraction": 0.2, "seed": int(seed)},
        "macro_roc_auc": float(macro_auc),
        "n_descriptors_scored": len(per_auc),
        "per_descriptor_auc": per_auc,
        "train_seconds": round(float(fit_seconds), 2),
        "fitted_on": "train_split",
        "model_random_state": int(seed),
    }
    (MODELS_DIR / "smell_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"macro ROC-AUC (held-out): {macro_auc:.3f} over {len(per_auc)} descriptors")
    print("saved models/smell.joblib")
    return metrics


def load_model() -> tuple[Any, list[str]]:
    """Load trained smell model and ordered 138 descriptor names."""
    model = joblib.load(MODELS_DIR / "smell.joblib")
    data = json.loads((MODELS_DIR / "descriptors.json").read_text(encoding="utf-8"))
    return model, data["names"]


def predict_proba(model: Any, smiles: list[str]) -> np.ndarray:
    """Predict (n, 138) descriptor probabilities for SMILES. Raises ValueError on invalid SMILES."""
    if len(smiles) == 0:
        n_outputs = getattr(model, "n_outputs_", 138)
        return np.empty((0, n_outputs), dtype=np.float64)
    X, kept = featurize(smiles)
    if len(kept) != len(smiles):
        kept_set = set(kept)
        bad_idx = next(i for i in range(len(smiles)) if i not in kept_set)
        raise ValueError(f"Unparsable SMILES at index {bad_idx}: {smiles[bad_idx]!r}")
    return _proba_matrix(model, X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OpenPOM RandomForest smell model.")
    parser.add_argument("--csv", default=None, help="Path to local OpenPOM CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    train(csv_path=args.csv, seed=args.seed)
