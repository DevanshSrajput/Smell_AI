# Backend Schema: Smell AI

There is no server and no database. The "backend" is a set of Python modules, JSON files, and joblib model files. This document fixes the contracts between them so each file can be built and tested alone.

## 1. Files on disk

| Path | Written by | Read by | Format |
|---|---|---|---|
| `data/curated_GS_LF_merged_4983.csv` | `smell_model.py` (download) | `smell_model.py` | CSV |
| `samples.json` | hand-authored | `sensor.py`, `train_nose.py`, `pipeline.py`, tests | JSON |
| `models/smell.joblib` | `smell_model.py` | `pipeline.py` | joblib |
| `models/descriptors.json` | `smell_model.py` | `pipeline.py`, tests | JSON |
| `models/smell_metrics.json` | `smell_model.py` | README, app sidebar (optional) | JSON |
| `models/sensor_bank.json` | `sensor.py` first use | `sensor.py`, `train_nose.py` | JSON |
| `models/nose.joblib` | `train_nose.py` | `pipeline.py` | joblib |
| `models/nose_metrics.json` | `train_nose.py` | README, tests | JSON |

## 2. `samples.json`

```json
{
  "version": 1,
  "samples": [
    {
      "id": "coffee",
      "label": "Coffee",
      "molecules": [
        {"name": "furfural", "smiles": "O=Cc1ccco1", "weight": 0.20}
      ]
    }
  ]
}
```

Rules:

- `id` is lowercase snake_case and unique. `label` is the display name.
- `weight` is a relative amount. Weights within one sample are normalized to sum to 1 at load time.
- Each sample has 5 to 8 molecules.
- Every `smiles` must parse in RDKit.

### Seed data

These are real aroma compounds associated with each sample. Validate every SMILES with RDKit before use and fix or drop any that fail. Weights are illustrative, not measured.

**coffee**: furfural `O=Cc1ccco1` 0.15; furfuryl mercaptan `SCc1ccco1` 0.10; guaiacol `COc1ccccc1O` 0.15; 4-ethylguaiacol `CCc1ccc(O)c(OC)c1` 0.10; 2,3-butanedione `CC(=O)C(C)=O` 0.15; 2,3-dimethylpyrazine `Cc1nccnc1C` 0.15; 5-methylfurfural `Cc1ccc(C=O)o1` 0.10; maltol `CC1=C(O)C(=O)C=CO1` 0.10

**vanilla_candle**: vanillin `COc1cc(C=O)ccc1O` 0.35; ethyl vanillin `CCOc1cc(C=O)ccc1O` 0.15; coumarin `O=c1ccc2ccccc2o1` 0.15; heliotropin `O=Cc1ccc2OCOc2c1` 0.15; benzyl benzoate `O=C(OCc1ccccc1)c1ccccc1` 0.20

**mint**: menthol `CC(C)C1CCC(C)CC1O` 0.35; menthone `CC(C)C1CCC(C)CC1=O` 0.20; carvone `CC(=C)C1CC=C(C)C(=O)C1` 0.10; 1,8-cineole `CC1(C)OC2CCC1(C)CC2` 0.10; pulegone `CC1CCC(=C(C)C)C(=O)C1` 0.10; limonene `CC(=C)C1CCC(C)=CC1` 0.15

**lemon**: limonene `CC(=C)C1CCC(C)=CC1` 0.45; citral `CC(C)=CCCC(C)=CC=O` 0.10; linalool `CC(C)=CCCC(C)(O)C=C` 0.10; beta-pinene `C=C1CCC2CC1C2(C)C` 0.10; gamma-terpinene `CC1=CCC(C(C)C)=CC1` 0.15; geranyl acetate `CC(=O)OCC=C(C)CCC=C(C)C` 0.10

**banana**: isoamyl acetate `CC(C)CCOC(C)=O` 0.50; isoamyl butyrate `CCCC(=O)OCCC(C)C` 0.15; butyl acetate `CCCCOC(C)=O` 0.15; hexyl acetate `CCCCCCOC(C)=O` 0.10; eugenol `COc1cc(CC=C)ccc1O` 0.10

**wood_smoke**: guaiacol `COc1ccccc1O` 0.25; 4-methylguaiacol `Cc1ccc(O)c(OC)c1` 0.15; syringol `COc1cccc(OC)c1O` 0.15; o-cresol `Cc1ccccc1O` 0.15; phenol `Oc1ccccc1` 0.10; furfural `O=Cc1ccco1` 0.10; isoeugenol `COc1cc(C=CC)ccc1O` 0.10

Coffee and wood smoke share guaiacol and furfural on purpose. They are the hardest pair for the nose. Report their confusion in the nose metrics.

## 3. `models/descriptors.json`

```json
{
  "names": ["alcoholic", "aldehydic", "..."],
  "base_rate": {"alcoholic": 0.031, "aldehydic": 0.044}
}
```

- `names` has length 138 and the same order as the model's output columns.
- `base_rate[name]` is the fraction of training molecules labelled positive for that descriptor.

## 4. `models/smell_metrics.json`

```json
{
  "n_molecules_total": 4983,
  "n_molecules_used": 4983,
  "n_dropped_unparsable": 0,
  "n_descriptors": 138,
  "featurizer": {"type": "morgan_count", "radius": 2, "n_bits": 2048},
  "model": {"type": "RandomForestClassifier", "n_estimators": 300, "max_depth": null, "min_samples_leaf": 1},
  "split": {"type": "random", "test_fraction": 0.2, "seed": 42},
  "macro_roc_auc": 0.0,
  "n_descriptors_scored": 0,
  "per_descriptor_auc": {"fruity": 0.0},
  "train_seconds": 0.0
}
```

`n_descriptors_scored` counts descriptors with both classes present in the test set. Macro AUC averages only those. Never leave an AUC value fabricated or hardcoded.

## 5. `models/sensor_bank.json`

```json
{
  "seed": 7,
  "n_sensors": 8,
  "feature_names": ["mw", "logp", "o_frac", "s_count", "n_count", "aromatic_rings", "tpsa", "hbd"],
  "feature_mean": [0.0],
  "feature_std": [1.0],
  "weights": [[0.0]],
  "bias": [0.0],
  "gain": [1.0],
  "humidity_coeff": [0.0],
  "drift_coeff": [0.0]
}
```

Shapes: `weights` is 8 by 8 (sensors by features). The other per-sensor lists have length 8. `feature_mean` and `feature_std` have length 8 and come from the 4,983 dataset molecules plus sample molecules, so standardization is stable.

## 6. `models/nose_metrics.json`

```json
{
  "n_sessions_per_sample": 200,
  "n_train": 0,
  "n_test": 0,
  "model": {"type": "RandomForestClassifier", "n_estimators": 200},
  "accuracy_default_noise": 0.0,
  "accuracy_by_noise": {"0.03": 0.0, "0.10": 0.0, "0.20": 0.0, "0.30": 0.0},
  "classes": ["coffee", "vanilla_candle", "mint", "lemon", "banana", "wood_smoke"],
  "confusion_matrix": [[0]],
  "seeds": {"sensor_bank": 7, "data": 1, "model": 1}
}
```

`accuracy_by_noise` is measured on fresh sessions generated at each noise level, using a model trained at the default noise range. This is the honest degradation curve.

## 7. Runtime contracts

### `sniff()` return value

```json
{
  "presented": "coffee",
  "conditions": {"noise": 0.03, "humidity": 0.5, "seed": null},
  "sensors": {"names": ["S1","S2","S3","S4","S5","S6","S7","S8"], "values": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
  "nose": {
    "predicted": "coffee",
    "confidence": 0.97,
    "probabilities": {"coffee": 0.97, "vanilla_candle": 0.0, "mint": 0.0, "lemon": 0.0, "banana": 0.01, "wood_smoke": 0.02},
    "correct": true
  },
  "description": [
    {"descriptor": "roasted", "score": 0.61, "lift": 8.2}
  ],
  "molecules": [
    {"name": "furfural", "smiles": "O=Cc1ccco1", "weight": 0.15}
  ]
}
```

- `description` has `top_k` items (default 5), sorted by the ranking rule in `ALGORITHMS.md` section 4.
- `score` is the weighted mean predicted probability across molecules.
- `lift` is `score / base_rate`.
- `molecules` is for the sample the nose predicted.

### Context object

```python
@dataclass
class Context:
    samples: dict[str, Sample]      # id -> Sample(label, molecules, weights normalized)
    bank: SensorBank
    nose: RandomForestClassifier
    nose_classes: list[str]
    smell_model: RandomForestClassifier
    descriptors: list[str]
    base_rate: np.ndarray           # shape (138,)
    mol_probs: dict[str, np.ndarray]  # sample id -> (n_molecules, 138), precomputed at load
```

Precompute `mol_probs` once in `load_all()` so each Sniff stays under the 300 ms budget.

## 8. Optional sniff log

Not required. If added, append one JSON line per sniff to `logs/sniffs.jsonl` using the `sniff()` return value plus an ISO timestamp. No database.
