# Smell AI — Evaluation Report (`eval.md`)

All numbers in this file come directly from measured runs recorded in:
- `models/smell_metrics.json` (CPU RandomForest baseline)
- `models/gnn_metrics.json` (GPU OpenPOM MPNN + Bemis-Murcko scaffold benchmark)
- `models/nose_metrics.json` (Simulated 8-channel electronic nose)
- `test_app.py` (7-check headless verification suite)

---

## 1. Overall Summary

| Component / Metric | Measured Value | Target | Source |
|---|---:|---:|---|
| **Dataset (`curated_GS_LF_merged_4983.csv`)** | 4,983 molecules, 0 dropped, 138 descriptors | 138 labels | `smell_metrics.json` |
| **RandomForest (CPU) — Random 80/20 Macro ROC-AUC** | **0.8317** (`138/138` scored) | ≥ 0.78 | `smell_metrics.json` |
| **OpenPOM MPNN (GPU) — Random 80/20 Macro ROC-AUC** | **0.8736** (`+0.0419` vs RF) | ≥ 0.78 | `gnn_metrics.json` |
| **RandomForest (CPU) — Bemis-Murcko Scaffold 80/20 AUC** | **0.7299** | — | `gnn_metrics.json` |
| **OpenPOM MPNN (GPU) — Bemis-Murcko Scaffold 80/20 AUC** | **0.7731** (`+0.0432` vs RF) | — | `gnn_metrics.json` |
| **Nose Classifier — Held-Out Accuracy (240 sessions)** | **96.7%** (`232/240`) | ≥ 95% | `nose_metrics.json` |
| **Nose Classifier — 20-Sniff Spot Check (`σ = 0.03`)** | **95.8%** (`115/120`, 5/6 samples ≥ 95%) | ≥ 95% | `test_app.py` |
| **Smell Model Fit Time (CPU RF / RTX 3050 GNN)** | `35.2 s` / `87.6 s` | < 3 min (RF) | `smell_metrics.json` / `gnn_metrics.json` |
| **Nose Model Train Time (CPU RF)** | `0.91 s` | < 60 s | `nose_metrics.json` |
| **Online `sniff()` Latency (50 sniffs)** | `~30 ms` mean, `44 ms` max | < 300 ms | local benchmark |

---

## 2. Smell Model Comparison: CPU RandomForest vs. GPU OpenPOM MPNN

| Property | RandomForest (CPU Baseline) | OpenPOM MPNN (GPU GNN) |
|---|---|---|
| **Script** | `smell_model.py` | `train_gnn.py` |
| **Input Representation** | 2048-bit Morgan count fingerprint (`radius=2`) | Atom graph (`31` node features) + Bond graph (`6` edge features) |
| **Architecture** | `RandomForestClassifier` (`300` trees, `balanced_subsample`) | `OpenPOMMPNN` (`hidden_dim=128`, `3` message steps, GRUCell, attentive + mean pooling, `256`-dim POM embedding, `476,299` params) |
| **Hardware** | Laptop CPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU (`cuda`) |
| **Random 80/20 Split (`seed=42`)** | `0.8317` | `0.8736` |
| **Bemis-Murcko Scaffold 80/20 Split** | `0.7299` | `0.7731` |
| **Saved Artifacts** | `models/smell.joblib`, `models/smell_metrics.json` | `models/smell_gnn.pt`, `models/gnn_metrics.json` |

### GNN Training Curve (Random 80/20 Split, `epochs=40`)

| Epoch | Train Loss | Val Macro ROC-AUC |
|---:|---:|---:|
| 5 | 0.5389 | 0.8627 |
| 10 | 0.4672 | 0.8729 |
| 15 | 0.4126 | **0.8736** |
| 20 | 0.3657 | 0.8724 |
| 25 | 0.3319 | 0.8718 |
| 30 | 0.3046 | 0.8692 |
| 35 | 0.2913 | 0.8693 |
| 40 | 0.2878 | 0.8693 |

### Best 10 & Worst 10 Per-Descriptor AUCs

#### RandomForest (CPU, `0.8317` macro)
- **Top 10:** `popcorn` (0.9962), `cabbage` (0.9744), `sulfurous` (0.9743), `ketonic` (0.9731), `cooling` (0.9685), `hazelnut` (0.9661), `bergamot` (0.9631), `cedar` (0.9631), `sandalwood` (0.9618), `smoky` (0.9609)
- **Worst 10:** `natural` (0.6926), `earthy` (0.6901), `berry` (0.6871), `dry` (0.6522), `sour` (0.6471), `grassy` (0.6468), `leafy` (0.6119), `musty` (0.6087), `cortex` (0.5764), `weedy` (0.3412)

#### OpenPOM MPNN (GPU, `0.8736` macro)
- **Top 10:** `ketonic` (0.9966), `popcorn` (0.9964), `cucumber` (0.9897), `terpenic` (0.9860), `sulfurous` (0.9818), `odorless` (0.9784), `black currant` (0.9778), `garlic` (0.9768), `onion` (0.9743), `hazelnut` (0.9726)
- **Worst 10:** `sweet` (0.7495), `celery` (0.7309), `leafy` (0.7299), `natural` (0.7266), `weedy` (0.7230), `grassy` (0.7076), `dry` (0.6954), `musty` (0.6943), `brandy` (0.6827), `tea` (0.5965)

---

## 3. Simulated Electronic Nose Evaluation (`models/nose_metrics.json`)

- **Training Setup:** `200` sessions per sample (`960` train / `240` test, 80/20 stratified split), noise $\sigma \sim \text{Uniform}(0.01, 0.06)$, humidity $\sim \text{Uniform}(0.2, 0.8)$, `RandomForestClassifier(n_estimators=200)`.
- **Seeds:** `sensor_bank=7`, `data=1`, `model=1`.
- **Held-out Accuracy (default noise range):** **96.7%** (`232/240`).

### Accuracy by Fixed Noise $\sigma$ (`300` fresh sessions/sample/level)

| Noise $\sigma$ | Accuracy |
|---:|---:|
| `0.03` | **98.7%** (`0.9872`) |
| `0.10` | **78.9%** (`0.7889`) |
| `0.20` | **53.5%** (`0.5350`) |
| `0.30` | **41.9%** (`0.4189`) |

### Confusion Matrix (240 Held-Out Sessions)

Rows = presented sample, Columns = predicted sample.

| | `coffee` | `vanilla_candle` | `mint` | `lemon` | `banana` | `wood_smoke` |
|---|---:|---:|---:|---:|---:|---:|
| **`coffee`** | 39 | 0 | 0 | 0 | 1 | 0 |
| **`vanilla_candle`** | 0 | 39 | 0 | 0 | 0 | 1 |
| **`mint`** | 0 | 0 | 40 | 0 | 0 | 0 |
| **`lemon`** | 0 | 0 | 3 | 37 | 0 | 0 |
| **`banana`** | 0 | 0 | 0 | 0 | 40 | 0 |
| **`wood_smoke`** | 0 | 3 | 0 | 0 | 0 | 37 |

- `coffee_wood_smoke_confusion`: `coffee_as_wood_smoke = 0`, `wood_smoke_as_coffee = 0`.
- Observed confusions: `lemon -> mint` (3), `wood_smoke -> vanilla_candle` (3), `coffee -> banana` (1), `vanilla_candle -> wood_smoke` (1).

---

## 4. Headless Verification (`test_app.py`, 7 Checks)

| Check | Result | Details |
|---|---|---|
| `[1/7]` Model files exist | PASS | All 6 required RF/nose files verified |
| `[2/7]` Sample SMILES parse | PASS | 6 samples, 37 molecules |
| `[3/7]` `sniff()` schema keys | PASS | All 6 samples |
| `[4/7]` 20-sniff spot check (`σ=0.03`) | PASS | `95.8%` overall (`coffee` 20/20, `vanilla_candle` 19/20, `mint` 19/20, `lemon` 19/20, `banana` 20/20, `wood_smoke` 18/20) |
| `[5/7]` Descriptor sanity (top 5) | PASS | `mint`, `lemon`, `banana`, `vanilla_candle`, `coffee`, `wood_smoke` |
| `[6/7]` Probabilities sum to 1 | PASS | Error `0.00e+00` |
| `[7/7]` Fixed-seed determinism | PASS | Exact match |
