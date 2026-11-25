# Deployment: Smell AI / SMELL//OS

The app is a single Streamlit entry point (`app.py`) backed by local Python modules and model artifacts. There is no server, database, or API key.

## 1. Local

Windows:

```powershell
uv venv .venv --python 3.12 --seed
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\python.exe src\smell_model.py
.venv\Scripts\python.exe src\train_nose.py
.venv\Scripts\python.exe tests\test_app.py
.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/smell_model.py
python src/train_nose.py
python tests/test_app.py
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 2. Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new app in Streamlit Cloud.
3. Set **Main file path** to `app.py`.
4. Select Python 3.12.
5. Add no secrets. The app needs no API keys.

### Model artifacts

`models/*.joblib` is gitignored. `smell.joblib` is about 255 MB compressed, which is not suitable for a normal GitHub repository. On first boot the app detects missing artifacts and shows an **INITIALIZE MODELS** button. That button:

1. Downloads the OpenPOM CSV to `data/`.
2. Trains and saves the smell model.
3. Trains and saves the nose model.
4. Clears the caches and reloads the UI.

First initialization takes roughly one to three minutes on a laptop-class CPU. Streamlit Cloud container storage is ephemeral: after a restart the app may need to initialize again. This is acceptable for a demo, not for a latency-sensitive production service.

If the host cannot download the CSV, run the training locally and host the model files somewhere the runtime can read, or build a Docker image (section 3). The dataset can also be supplied with `python src/smell_model.py --csv PATH`.

### Memory note

The smell model is a 300-tree multi-output RandomForest with 138 outputs. It is large in memory as well as on disk. A host with at least 2 GB of RAM is recommended. If a free-tier host runs out of memory, the options are:

- Lower `n_estimators` in `smell_model.py` and retrain, then report the new measured AUC honestly.
- Host the app on a larger instance.
- Replace the forest with a smaller model, keeping the `predict_proba(model, smiles) -> (n, 138)` contract.

Do not commit a smaller model while claiming the metrics of the larger one.

### Health check

Streamlit exposes `/_stcore/health`:

```bash
curl http://localhost:8501/_stcore/health
```

## 3. Docker

The included `Dockerfile` installs the requirements and trains both models at build time.

```bash
docker build -t smell-ai .
docker run --rm -p 8501:8501 smell-ai
```

To keep artifacts and logs between restarts:

```bash
docker run --rm -p 8501:8501 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/logs:/app/logs" \
  smell-ai
```

On PowerShell, use `${PWD}`:

```powershell
docker run --rm -p 8501:8501 `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/logs:/app/logs" `
  smell-ai
```

If the build host cannot reach GitHub, place `data/curated_GS_LF_merged_4983.csv` in the build context before building. The Dockerfile does not ignore `data/`, so a local copy is used.

## 4. What to verify after deploying

- The hero shows `MODELS: LOADED` and the sidebar shows the measured AUC and accuracy.
- **Sniff Lab** returns sensor channels, a nose guess, probabilities, and five descriptors.
- **Model Metrics** renders the confusion matrix and noise curve from `models/*.json`.
- **Documentation** tabs render the markdown files in `docs/`.
- `logs/sniffs.jsonl` grows after a single sniff. Batch and sweep runs do not write to the log.

## 5. Environment variables

None are required. Optional values for a custom deployment:

| Variable | Purpose |
|---|---|
| `STREAMLIT_SERVER_PORT` | Override the server port. |
| `STREAMLIT_SERVER_ADDRESS` | Bind address, usually `0.0.0.0` in containers. |

No secrets, tokens, or API keys are used by this project.
