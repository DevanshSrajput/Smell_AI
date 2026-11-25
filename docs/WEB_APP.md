# SMELL//OS Web App Guide

SMELL//OS is the Cybercore Streamlit interface for the Smell AI pipeline. It exposes every backend capability: single sniffs, repeated batches, single-molecule analysis, environmental sweeps, model metrics, the sample library, the sniff log, and the full documentation set.

## 1. Global controls (sidebar)

| Control | Meaning |
|---|---|
| **Sample** | Which of the six samples is presented to the nose. |
| **Noise σ** | Gaussian noise added to each of the eight simulated channels. Range 0.00–0.30. |
| **Humidity** | Shifts the sensor baseline. Range 0.0–1.0. |
| **Seed** | Blank gives a fresh reading each press. An integer makes the sniff repeatable. |

The sidebar also shows the measured smell-model AUC, nose accuracy, sensor-bank seed, and runtime status. **RESET SESSION** clears the last sniff, batch, molecule result, and sweep without touching the models.

## 2. Sniff Lab

The primary flow from the PRD.

1. Pick a smell in the sidebar.
2. Set noise, humidity, and optionally a seed.
3. Press **INITIATE SNIFF**.
4. Read the four result rows:
   - eight sensor channels relative to clean air;
   - the nose's guess and confidence, plus all six probabilities;
   - the top five descriptors for the **predicted** sample;
   - the molecules behind that description.

If the nose is wrong, the UI says so and the description follows the guess. The **INFERENCE TRACE** expander contains the complete `sniff()` return value. **DOWNLOAD SNIFF JSON** saves it.

## 3. Batch Arena

Runs repeated sniffs for every sample under the current sidebar conditions and reports:

- overall accuracy and mean confidence;
- per-sample accuracy;
- a 6×6 confusion matrix;
- the most frequent top descriptor;
- a downloadable CSV of every sniff.

Repeats are configurable from 5 to 30 per sample. Batch runs do not write to the sniff log.

## 4. Molecule Lab

Analyzes one SMILES string with the smell model:

- top 12 descriptors with score and lift;
- the eight simulator properties and their z-scores;
- a **nose resemblance** chart: the nose is run on a one-molecule "session" to show which sample class it resembles.

Resemblance is not identification. The nose was trained on six mixtures, not single molecules.

## 5. Sensor Sweep

Sweeps one environmental axis and plots the nose's accuracy and mean confidence:

- **Noise σ:** 0.00 → 0.30, humidity fixed at the sidebar value.
- **Humidity:** 0.00 → 1.00, noise fixed at the sidebar value.

Levels and repeats are configurable. A sweep writes a CSV with one row per sniff. Higher settings take longer; each sniff is about 30 ms locally.

## 6. Model Metrics

Reads the committed JSON metrics; nothing is hardcoded.

- Smell model: macro ROC-AUC, descriptor coverage, featurizer, model, split, per-descriptor AUC with a filter and CSV download.
- Nose model: held-out accuracy, confusion matrix, noise-degradation curve, seeds, and the explicit coffee/wood-smoke confusion counts.

## 7. Sample Library

Shows the six hand-authored compositions, their molecule tables with RDKit properties, composition weights, and the top-10 descriptor profile for the selected sample. `samples.json` can be downloaded.

## 8. Sniff Log

Reads the last 500 lines of `logs/sniffs.jsonl`. Shows timestamp, presented sample, predicted sample, confidence, verdict, noise, humidity, and seed. Filter by sample and download CSV or the last 500 JSONL records.

The log is runtime state. It is gitignored and does not exist until the first sniff.

## 9. Documentation

Most documentation tabs are rendered from the repository markdown; **API** and **HONESTY** are maintained inline in `app.py`:

- **GUIDE** — this file;
- **QUICKSTART** — `README.md`;
- **ARCHITECTURE** — `docs/TRD.md`;
- **ALGORITHMS** — `docs/ALGORITHMS.md`;
- **SCHEMAS** — `docs/BACKEND_SCHEMA.md`;
- **APP FLOW** — `docs/APP_FLOW.md`;
- **PRODUCT** — `docs/PRD.md`;
- **DEPLOYMENT** — `docs/DEPLOYMENT.md`;
- **API** — Python usage examples;
- **HONESTY** — what is real, what is simulated, and what the numbers mean.

## 10. Keyboard-free workflow

The app is intentionally button-driven:

- changing a control does not rerun a sniff;
- a sniff uses the values at the moment the button is pressed;
- a fixed seed reproduces the full result;
- a blank seed produces a new reading on every press.

## 11. Performance

`pipeline.load_all()` is cached with `st.cache_resource`. Molecule probabilities for all 37 sample molecules are precomputed once. A single sniff is about 30 ms locally; the dominant cost is the first model load. Batch and sweep runs reuse the cached context.
