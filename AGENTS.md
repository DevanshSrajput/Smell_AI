# AGENTS.md — Smell AI

## Project Overview
Local Python + Streamlit app joining a smell model (SMILES -> 138 OpenPOM odor descriptors via RandomForest) with a simulated 8-channel electronic nose (8 RDKit properties -> 6 samples). Runs on laptop CPU; no GPU, no hardware, no API keys.

## File Layout (`TRD.md` §4, organized)
```
smell-ai/
  data/samples.json                    # tracked; *.csv downloaded & gitignored
  models/                              # *.joblib & *.pt gitignored; *.json committed
    smell.joblib | descriptors.json | smell_metrics.json
    smell_gnn.pt | gnn_metrics.json
    sensor_bank.json | nose.joblib | nose_metrics.json
  logs/sniffs.jsonl                    # runtime sniff log, gitignored
  src/                                 # smell_model.py, train_gnn.py, sensor.py, train_nose.py, pipeline.py
  tests/test_app.py                    # 7-check headless suite
  docs/                                # specs, PROJECT_SKETCH.md, eval.md
  app.py, requirements.txt, requirements-gpu.txt, Dockerfile, README.md
```

## Run Commands (`APP_FLOW.md` §1)
Windows `.venv` note: execute exclusively via `.venv\Scripts\python.exe` and `.venv\Scripts\pip.exe` (never global Python/pip).
1. `uv venv .venv --python 3.12 --seed && .venv\Scripts\pip.exe install -r requirements.txt`
2. `.venv\Scripts\python.exe src\smell_model.py` (downloads OpenPOM CSV, trains, saves `smell.joblib`)
3. `.venv\Scripts\python.exe src\train_nose.py` (trains nose, saves `nose.joblib` & metrics)
4. `.venv\Scripts\python.exe tests\test_app.py` (7-check headless suite, exits 0)
5. `.venv\Scripts\python.exe -m streamlit run app.py`

## Ground Rules
- Environment: Python 3.10-3.12 CPU-only `.venv`. No `torch`, `deepchem`, GPU libs, API keys, or frameworks beyond Streamlit.
- Style & Scope: Work only in `c:\Personal Projects\smell-ai`. Follow Ponytail (full intensity: YAGNI, stdlib/installed deps first, `# ponytail:` on deliberate simplifications). Keep code plain and short; comment only non-obvious intent.
- Integrity: Never fabricate or hardcode metrics. Assert 138 OpenPOM label columns; never loosen test thresholds. Keep simulated-nose honesty in UI footer and README.
- Precedence: `BACKEND_SCHEMA.md` (shapes) > `ALGORITHMS.md` (math) > `TRD.md` (interfaces); report doc conflicts.
- Git: One commit per phase/logical step with a single-line commit message.

## Definition of Done
- `smell_model.py` asserts 138 labels, runs < 3 min, achieves >= 0.78 measured macro ROC-AUC.
- `train_nose.py` runs < 60 s, achieves >= 95% default-noise accuracy, saves confusion matrix + noise curve.
- `pipeline.py` runs `sniff()` < 300 ms and logs to `logs/sniffs.jsonl`; `test_app.py` exits 0.
- `app.py` runs cleanly (Pick-then-Sniff + Single molecule tab + simulated-nose footer); `README.md` uses measured JSON metrics only.
