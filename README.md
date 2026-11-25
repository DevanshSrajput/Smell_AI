# Smell AI

A local Python + Streamlit app that joins two models:

1. **Smell model** — SMILES → 138 OpenPOM odor descriptors (RandomForest on Morgan count fingerprints).
2. **Simulated electronic nose** — 8 RDKit-derived "sensor" channels → one of six samples (RandomForest).

Pick a smell, press **Sniff**, and the app shows the sensor traces, the nose's guess and confidence, and the smell model's description of the molecules behind that guess. Everything runs on a laptop CPU. No GPU, no hardware, no API keys.

> **Honesty first:** the nose is simulated. Its eight channels are computed from molecular properties, not measured by gas sensors. Its high accuracy is measured against the same simulator that generated the training data, so it says nothing about real hardware. The molecule model predicts descriptor probabilities from structure; those are model scores, not measurements of a real smell.

## Quickstart

Windows (PowerShell):

```powershell
uv venv .venv --python 3.12 --seed
.venv\Scripts\pip.exe install -r requirements.txt

.venv\Scripts\python.exe smell_model.py   # downloads OpenPOM CSV, trains, prints macro ROC-AUC
.venv\Scripts\python.exe train_nose.py    # trains the nose, prints accuracy and confusion matrix
.venv\Scripts\python.exe test_app.py      # 7-check headless suite, exits 0
.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python smell_model.py
python train_nose.py
python test_app.py
streamlit run app.py
```

The first `smell_model.py` run downloads the 4,983-molecule CSV from the [OpenPOM repository](https://github.com/BioMachineLearning/openpom) to `data/`. If that URL is unavailable, download the file yourself and run:

```bash
python smell_model.py --csv path/to/curated_GS_LF_merged_4983.csv
```

Python 3.10–3.12, CPU only. `requirements.txt` is deliberately small: RDKit, scikit-learn, pandas, numpy, joblib, Streamlit, requests.

## Web app (SMELL//OS)

`app.py` is a minimal light Y2K Streamlit console with eight tabs:

| Tab | What it does |
|---|---|
| **Sniff Lab** | Pick a smell, sniff, and read the sensor array, nose guess, probabilities, top descriptors, molecules, and a downloadable JSON trace. |
| **Batch Arena** | Repeated sniffs across all six samples, with accuracy, confidence, a confusion matrix, and CSV export. |
| **Molecule Lab** | Analyze one SMILES: top descriptors, eight sensor properties, and a nose-resemblance chart. |
| **Sensor Sweep** | Sweep noise or humidity and plot accuracy and confidence degradation. |
| **Model Metrics** | Measured smell AUC, per-descriptor AUC, nose confusion matrix, noise curve, and seeds. |
| **Sample Library** | The six compositions, molecule properties, weights, and descriptor profiles. |
| **Sniff Log** | The last 500 sniffs from `logs/sniffs.jsonl`, filterable and downloadable. |
| **Documentation** | Ten documentation tabs for the guide, specs, deployment, API, and honesty; most render `README.md` and `docs/`. |

Global controls live in the sidebar: sample, noise, humidity, and seed. A fixed seed reproduces a sniff; a blank seed gives a fresh reading. Batch and sweep runs do not write to the sniff log.

### Deploying

Local:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

Streamlit Community Cloud: push the repository, set the main file to `app.py`, choose Python 3.12, and add no secrets. Model joblib files are gitignored and too large for a normal repository, so the app detects missing artifacts and offers an **INITIALIZE MODELS** button that downloads the OpenPOM CSV and trains both models in the runtime. First boot takes about one to three minutes. Container storage is ephemeral, so a restart may retrain.

A `Dockerfile` is included for a self-contained image that trains at build time. Full instructions, memory notes, and health checks are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). The tab-by-tab user guide is in [`docs/WEB_APP.md`](docs/WEB_APP.md).

## How it works

```
OpenPOM CSV ──> smell_model.py ──> models/smell.joblib + descriptors.json
samples.json ─> train_nose.py ───> models/nose.joblib
                     ^
sensor.py (simulator)┘

app / test_app
  └─ pipeline.sniff(ctx, sample, noise, humidity, seed)
       ├─ sensor.simulate(...)           8 readings from the sample's molecules
       ├─ nose.predict_proba(...)        predicted sample + confidence
       ├─ describe(ctx, predicted)       top-5 descriptors, score × sqrt(lift)
       └─ returns sensors, nose, description, molecules
```

The **predicted** sample selects the molecules used for the description, not the presented one. If the nose is wrong, the description is wrong in the same way, and the UI says so. The smell model's probabilities for every sample molecule are precomputed once in `pipeline.load_all()`, so a sniff does not rerun the molecule model.

### Smell model

- Morgan count fingerprints, radius 2, 2048 bits.
- One multi-output `RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample")` over all 138 labels.
- Random 80/20 split, seed 42. The saved model is fitted on the 80% train split so it matches the reported metric.
- Descriptors are ranked by `score × sqrt(lift)`, where `lift = score / max(base_rate, 0.005)`. This is the `ALGORITHMS.md` §4 rule, applied to all six samples. Without the lift term, common descriptors such as `sweet` or `fruity` dominate every sample.

### Simulated nose

Each molecule becomes 8 properties: molecular weight, Crippen logP, oxygen fraction, sulfur count, nitrogen count, aromatic rings, TPSA, and HBD. A fixed seeded sensor bank (seed 7) maps standardized properties through sparse random weights, then applies gain, presentation intensity, volatility, saturation, humidity, drift, and Gaussian noise. The bank is stored in `models/sensor_bank.json` so training and serving use the same sensors.

The nose is a 200-tree RandomForest on the raw 8-vector. Training data: 200 sessions per sample, noise σ ~ Uniform(0.01, 0.06), humidity ~ Uniform(0.2, 0.8), split 80/20 stratified by sample. Test sessions are never used in training.

## Measured results

These numbers come from the committed JSON metrics files (`models/smell_metrics.json`, `models/gnn_metrics.json`, `models/nose_metrics.json`).

| Metric | Measured | Target |
|---|---:|---:|
| Smell model (RandomForest, Random 80/20) macro ROC-AUC | **0.832** | ≥ 0.78 |
| Smell model (OpenPOM MPNN GNN, Random 80/20) macro ROC-AUC | **0.874** | ≥ 0.78 |
| Smell model (Bemis-Murcko Scaffold 80/20): GNN vs. RF | **0.773** vs. 0.730 | — |
| Descriptors scored | 138 of 138 | 138 |
| Nose held-out accuracy (240 sessions) | **96.7%** (232/240) | ≥ 95% |
| `sniff()` latency (local, 50 sniffs) | **~30 ms** mean, 44 ms max | < 300 ms |
| Smell model fit (`train_seconds`, CPU RF / RTX 3050 GNN) | 35.2 s / 87.6 s | — |
| Nose train (`train_seconds`) | 0.9 s | < 60 s |

Nose accuracy by fixed noise σ (300 fresh sessions per sample per level):

| Noise σ | Accuracy |
|---:|---:|
| 0.03 | 98.7% |
| 0.10 | 78.9% |
| 0.20 | 53.5% |
| 0.30 | 41.9% |

Confusion matrix on the 240 held-out sessions (rows = presented, columns = predicted):

| | coffee | vanilla | mint | lemon | banana | wood smoke |
|---|---:|---:|---:|---:|---:|---:|
| **coffee** | 39 | 0 | 0 | 0 | 1 | 0 |
| **vanilla** | 0 | 39 | 0 | 0 | 0 | 1 |
| **mint** | 0 | 0 | 40 | 0 | 0 | 0 |
| **lemon** | 0 | 0 | 3 | 37 | 0 | 0 |
| **banana** | 0 | 0 | 0 | 0 | 40 | 0 |
| **wood smoke** | 0 | 3 | 0 | 0 | 0 | 37 |

The design docs expected coffee and wood smoke to be the hardest pair because they share guaiacol and furfural. In this trained model the observed confusions were **lemon → mint** and **wood smoke ↔ vanilla candle**; coffee and wood smoke were not confused in these 240 sessions. The matrix above is the measured result, and `nose_metrics.json` records that pair explicitly (`coffee_wood_smoke_confusion`: 0 and 0).

### What each sample smells like (actual top 5)

| Sample | Top descriptors |
|---|---|
| Coffee | caramellic, sweet, phenolic, coffee, spicy |
| Vanilla Candle | vanilla, creamy, sweet, chocolate, hay |
| Mint | mint, herbal, fresh, woody, pine |
| Lemon | terpenic, lemon, citrus, orange, fresh |
| Banana | banana, pear, solvent, ripe, fruity |
| Wood Smoke | phenolic, medicinal, leathery, vanilla, smoky |

### Why the nose number is optimistic

Train and test sessions come from the same simulator. A 96.7% held-out score shows the pipeline and the classifier work end to end; it does **not** estimate accuracy on a real BME688. The noise curve above is the honest part: as the slider moves from 0.03 to 0.30, accuracy falls from 98.7% to 41.9%.

The smell model uses a random split, not a scaffold split. Its 0.832 macro ROC-AUC may therefore be optimistic for molecules with scaffolds unlike anything in training. The OpenPOM authors report a higher number with a graph neural network (see Part 4); this project deliberately stays with a fingerprint forest for CPU-only reproducibility.

## Project layout

```
smell-ai/
  data/curated_GS_LF_merged_4983.csv   downloaded, gitignored
  models/
    smell.joblib | descriptors.json | smell_metrics.json
    sensor_bank.json | nose.joblib | nose_metrics.json
  logs/sniffs.jsonl                    runtime sniff log, gitignored
  samples.json                         six samples, 37 molecules
  requirements.txt
  .streamlit/config.toml                light Y2K theme
  Dockerfile / .dockerignore            self-contained deployment
  PROJECT_SKETCH.md                     architecture sketch and module map
  smell_model.py                        train/serve the 138-descriptor model
  sensor.py                             simulated 8-channel nose
  train_nose.py                         generate sessions, train nose
  pipeline.py                           load_all(), sniff(), describe()
  app.py                                light Y2K Streamlit UI (8 tabs)
  test_app.py                           7-check headless suite
  docs/                                 PRD, TRD, APP_FLOW, BACKEND_SCHEMA, ALGORITHMS,
                                        DEPLOYMENT, WEB_APP
```

`sniff()` appends one JSON line per sniff to `logs/sniffs.jsonl` with an ISO timestamp by default; batch and sweep calls pass `log=False`. There is no database.

## Tests

`python test_app.py` runs seven checks:

1. All model files exist; otherwise it prints the command that creates each missing file.
2. Every SMILES in `samples.json` parses in RDKit.
3. `sniff()` returns the documented keys for all six samples.
4. Nose accuracy over 20 sniffs per sample at noise 0.03: at least 5 of 6 samples must reach 95%, and overall accuracy must be at least 95%.
5. Descriptor sanity: each sample's top 5 must contain at least one expected term (mint/cooling/herbal, citrus/lemon/fresh/terpenic, and so on). Expected names are filtered against `descriptors.json`; an empty expected set is a failure.
6. Nose probabilities sum to 1 within 1e-6.
7. The same seed produces the same sensor vector, prediction, and description.

Current run: **PASS**, 95.8% overall in the fixed-seed 20-sniff spot check. `wood_smoke` scored 18/20 (90%) in that spot check, which is allowed by the 5-of-6 rule; the larger 240-session held-out evaluation is 96.7%.

No threshold is loosened to make a test pass. If a descriptor check fails, the test prints the actual top 5.

## Reproducibility

| Component | Seed |
|---|---:|
| Smell model train/test split and RandomForest | 42 |
| Sensor bank | 7 (recorded in `sensor_bank.json`) |
| Nose data generation | 1 |
| Nose RandomForest | 1 |

All seeds are recorded in the metrics JSON. The sensor bank is generated once and loaded from disk afterwards; the app never regenerates it with a different seed.

## Give it a real nose (Part 3)

The simulator can be replaced without touching the rest of the pipeline:

- **Hardware:** BME688 on an ESP32 over I2C.
- **Heater steps:** 200, 250, 300, 350 °C. At each step read gas resistance and divide by the clean-air reading; concatenate across steps into one feature row.
- **Protocol:** 24-hour burn-in. At least three sessions on different days. Test on a session never seen in training. Split by session, not by row, so no session leaks across train and test.
- **Samples:** use room-temperature coffee grounds, not hot coffee (hot coffee mostly teaches the sensor to detect steam). Record each smell in its own jar for 30 minutes with clean air between. Start with very different smells.
- **Code:** replace `sensor.simulate` with a CSV loader that returns the same 8-vector shape (or 4 heater steps × however many features), then retrain with `train_nose.py`. Keep the session-level split.

The simulator's noise curve is a reminder that real sensors drift and respond to humidity; a real deployment needs calibration and much more data.

## Upgrade path: OpenPOM MPNN (GNN)

`train_gnn.py` trains an edge-conditioned Message Passing Neural Network (MPNN) with attentive pooling on atom/bond graphs using PyTorch CUDA (measured on an NVIDIA RTX 3050 6GB Laptop GPU: **0.8736** macro ROC-AUC on the random 80/20 split and **0.7731** on the Bemis-Murcko scaffold split, compared to `0.8317` and `0.7299` for the CPU RandomForest baseline). When `models/smell_gnn.pt` is present, `pipeline.py` and `app.py` let you switch between the two models in the sidebar and compare them side-by-side in Molecule Lab and Model Metrics.

## Data, documentation, and credit

- Architecture sketch: [`PROJECT_SKETCH.md`](PROJECT_SKETCH.md)
- Specifications and guides: [`docs/PRD.md`](docs/PRD.md), [`docs/TRD.md`](docs/TRD.md), [`docs/BACKEND_SCHEMA.md`](docs/BACKEND_SCHEMA.md), [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md), [`docs/APP_FLOW.md`](docs/APP_FLOW.md), [`docs/WEB_APP.md`](docs/WEB_APP.md), [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
- Dataset: `curated_GS_LF_merged_4983.csv` from the [OpenPOM repository](https://github.com/BioMachineLearning/openpom), derived from Lee et al., *Science* (2023), "A principal odor map unifies diverse tasks in olfactory perception." Check the OpenPOM repository for dataset licensing before redistribution.
- Descriptor labels are the 138 OpenPOM odor descriptors.
- The six sample recipes are illustrative aroma compositions, not measured formulations. Weights are relative amounts only.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing model files` in the app | Run `python smell_model.py` and `python train_nose.py`. |
| Dataset download fails | Download the CSV and run `python smell_model.py --csv PATH`. |
| RDKit install fails | Use the PyPI wheels (no compiler needed) or a conda environment; Python 3.10–3.12. |
| Nose looks wrong at high noise | Expected. Move the noise slider back toward 0.03; the degradation curve is in the metrics. |
| Want a repeatable sniff | Enter an integer seed in the sidebar; blank means a fresh reading each press. |

This is a portfolio/learning project. Do not use it for safety, medical, or fragrance-formulation decisions.
