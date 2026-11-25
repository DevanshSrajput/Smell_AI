# TRD: Smell AI with a Simulated Electronic Nose

## 1. Stack

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.10 to 3.12 | Matches the reference write-up. |
| Chemistry | RDKit (PyPI `rdkit`) | SMILES parsing, Morgan fingerprints, molecular descriptors. |
| ML | scikit-learn | Random Forest for both models. CPU only, fast. |
| Data | pandas, numpy | CSV and arrays. |
| Persistence | joblib, json | Model files and metrics. |
| UI | Streamlit | Matches the reference. Single file app. |
| Charts | Streamlit native (`st.bar_chart`) or Plotly | Keep dependencies small. Native first. |

`requirements.txt` (pin loosely, test on a clean venv):

```
rdkit>=2023.9
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
joblib>=1.3
streamlit>=1.30
requests>=2.31
```

No torch, no deepchem, no GPU libraries.

## 2. Architecture

```
            OFFLINE (run once)                              ONLINE (per Sniff)

 OpenPOM CSV --> smell_model.py --> models/smell.joblib ----+
                                    models/descriptors.json |
                                                            v
 samples.json --> train_nose.py --> models/nose.joblib --> pipeline.sniff() --> app.py
                      ^                                     ^   ^
                      |                                     |   |
                  sensor.py (simulator) ---------------------+   |
                  samples.json (composition lookup) ------------+
```

Data flow inside `sniff()`:

1. Look up the sample's molecules in `samples.json`.
2. `sensor.simulate()` turns molecules plus conditions into an 8-vector (R/R0 style).
3. The nose model maps the 8-vector to a sample label and probabilities.
4. The predicted label (not the true label) selects a composition from `samples.json`.
5. The smell model scores each molecule. A weighted aggregation gives sample-level descriptor scores.
6. Return everything as one dict.

Step 4 uses the predicted label on purpose. If the nose is wrong, the description is wrong too, and the UI shows that. This is how the reference pipeline works: the nose identifies, then composition is looked up.

## 3. Modules

### 3.1 `smell_model.py`

Public functions:

```python
DATA_URL = "https://raw.githubusercontent.com/BioMachineLearning/openpom/main/openpom/data/curated_GS_LF_merged_4983.csv"

def load_dataset(csv_path: str | None = None) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (smiles, Y[n, 138], descriptor_names). Downloads to data/ if missing."""

def featurize(smiles: list[str]) -> tuple[np.ndarray, list[int]]:
    """Morgan count fingerprints. Returns (X, kept_indices). Drops unparsable SMILES."""

def train(csv_path=None, seed=42) -> dict:
    """Train, evaluate, save. Returns metrics dict."""

def load_model() -> tuple[Model, list[str]]
def predict_proba(model, smiles: list[str]) -> np.ndarray   # shape (n, 138)
```

CLI: `python smell_model.py [--csv PATH] [--seed 42]`.

Dataset note: the CSV has a SMILES column named `nonStereoSMILES`, a free-text descriptors column, and 138 binary label columns. The code must not hardcode column positions. Identify the SMILES column by name, then treat every remaining column whose values are all in {0, 1} as a label column. Print the count and assert it is 138. If the count differs, stop and print what was found instead of guessing.

Outputs:

- `models/smell.joblib`
- `models/descriptors.json`: ordered list of 138 names plus per-descriptor base rate (fraction of positive molecules)
- `models/smell_metrics.json`: see `BACKEND_SCHEMA.md`

### 3.2 `sensor.py`

Public functions:

```python
N_SENSORS = 8
SENSOR_NAMES = ["S1", ..., "S8"]

def molecule_features(smiles: str) -> np.ndarray:
    """Property vector for one molecule. See ALGORITHMS.md section 2."""

def make_sensor_bank(seed=7) -> SensorBank:
    """Fixed random-but-seeded weights. Same bank every run."""

def simulate(molecules: list[dict], noise: float, humidity: float, rng) -> np.ndarray:
    """Return shape (8,) readings."""
```

The bank is generated once from a fixed seed and written to `models/sensor_bank.json` so train and app use identical sensors. Do not regenerate it at app start with a different seed.

### 3.3 `train_nose.py`

```python
def build_dataset(n_sessions=200, seed=1) -> tuple[X, y, groups]
def train(seed=1) -> dict
```

CLI: `python train_nose.py`. Writes `models/nose.joblib` and `models/nose_metrics.json`.

### 3.4 `pipeline.py`

```python
def load_all() -> Context              # loads samples, bank, nose, smell model, descriptors
def sniff(ctx, sample_name: str, noise=0.03, humidity=0.5, seed=None) -> dict
def describe(ctx, sample_name: str, top_k=5) -> list[dict]
```

`pipeline.py` has no Streamlit imports. The app and the tests both call it.

### 3.5 `app.py`

Streamlit. Caches `load_all()` with `st.cache_resource`. Reads nothing from disk per click except through the cached context. See `APP_FLOW.md`.

### 3.6 `test_app.py`

Plain script, no pytest required (pytest compatible). Details in section 6.

## 4. Repository layout at runtime

```
smell-ai/
  data/curated_GS_LF_merged_4983.csv      (downloaded, gitignored)
  models/
    smell.joblib
    descriptors.json
    smell_metrics.json
    sensor_bank.json
    nose.joblib
    nose_metrics.json
  samples.json
  *.py
```

`.gitignore`: `.venv/`, `data/`, `models/*.joblib`, `__pycache__/`.

## 5. Performance budget

| Step | Budget |
|---|---|
| Featurize 4,983 molecules | under 10 s |
| Train smell model | under 2 min on 4 cores |
| Generate 1,200 sniff sessions | under 5 s |
| Train nose | under 30 s |
| One Sniff in the app | under 300 ms |

If the smell model exceeds budget, lower `n_estimators` or `max_depth` before changing the model family. Record the choice in `smell_metrics.json`.

## 6. Test plan

`test_app.py` checks, in order:

1. All model files exist. If not, print which command to run and exit 1.
2. Every SMILES in `samples.json` parses in RDKit.
3. `sniff()` returns the documented keys for all six samples.
4. With default noise and fixed seeds, the nose gets at least 5 of 6 correct on 20 sniffs each (aim 100%).
5. Descriptor sanity, soft-expected sets:
   - mint: any of `mint`, `cooling`, `herbal`, `camphoreous`, `minty`
   - lemon: any of `citrus`, `lemon`, `fresh`, `terpenic`
   - banana: any of `fruity`, `banana`, `ester`, `sweet`
   - vanilla candle: any of `vanilla`, `sweet`, `creamy`, `balsamic`
   - coffee: any of `roasted`, `coffee`, `nutty`, `caramellic`
   - wood smoke: any of `smoky`, `phenolic`, `woody`, `burnt`
   Descriptor names must come from the actual list in `descriptors.json`. Filter the expected sets to names that exist, and fail if a set ends up empty.
6. The nose probabilities sum to 1 within 1e-6.
7. Determinism: same seed gives the same reading vector.

On any failure, print the actual top 5 descriptors or confusion row so the cause is visible. Do not loosen a threshold to make a test pass without telling the user.

## 7. Error handling

- Missing model files: app shows `st.error` with the two training commands and stops.
- Download failure: `smell_model.py` prints the URL, the target path, and the `--csv` option, then exits 1.
- Unparsable SMILES in the dataset: drop and count. Print the count.
- Unparsable SMILES in `samples.json`: fail loudly with the molecule name.

## 8. Reproducibility

Fixed seeds everywhere: dataset split 42, sensor bank 7, nose data 1, Random Forest `random_state` set. All seeds recorded in the metrics JSON.

## 9. Upgrade paths (documented, not built)

- Smell model: swap in OpenPOM's MPNN (Python 3.10, PyTorch, DeepChem, GPU helps). Keep the same `predict_proba` signature so nothing else changes.
- Real nose: BME688 on ESP32 over I2C, heater steps 200/250/300/350 C, gas resistance per step divided by the clean-air reading, 24 hour burn-in, at least three sessions on different days, test on a session never seen in training. Replace `sensor.simulate` with a CSV loader that returns the same 8-vector shape (or 4 steps times however many features), and retrain with `train_nose.py`.
