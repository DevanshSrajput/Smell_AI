# PRD: Smell AI with a Simulated Electronic Nose

## 1. What this is

A local Python app with two models chained together. A smell model reads a molecule and predicts which of 138 odor descriptors apply to it (fruity, minty, smoky and so on). A simulated electronic nose reads 8 fake gas sensor channels and decides which of six real-world samples it is smelling. The app joins them: the nose names the sample, the sample's known molecules go to the smell model, and the app shows the smell in words.

Everything runs on a laptop CPU in a few minutes. No GPU, no hardware, no API keys.

Reference project: "Build an AI that can smell" (the attached write-up). Dataset and research lineage: OpenPOM (https://github.com/BioMachineLearning/openpom), 4,983 molecules labelled with 138 descriptors, from the Lee et al. 2023 Science paper on the principal odor map.

## 2. Who it is for

Devansh, as a portfolio and learning project, and anyone who clones the repo afterwards. The reader of the README should be able to run it in five commands and understand the pipeline in ten minutes.

## 3. Goals

1. One command trains the smell model on the OpenPOM dataset and saves it.
2. One command trains the nose on simulated sniffs and saves it.
3. A Streamlit app lets the user pick a smell, press Sniff, and see sensor channels, the nose's guess with confidence, and the smell model's description.
4. A headless test script confirms the whole pipeline works without a browser.
5. The write-up is honest about what is simulated and what is not.

## 4. Non-goals

- No real hardware code in v1. The BME688 + ESP32 route stays a documented extension (README, "Give it a real nose").
- No graph neural network in v1. OpenPOM's GNN needs PyTorch and DeepChem. It is a documented upgrade path only.
- No user accounts, database, or deployment.
- No claim that the app can smell. It classifies simulated signals and predicts descriptors from structure.

## 5. Users and primary flow

The user opens the app, picks one of six smells from a dropdown, presses **Sniff**, and sees:

- 8 sensor channel bars reacting
- the nose's top guess and its confidence, plus probabilities for all six
- the top 5 smell descriptors for that sample, with scores
- the list of molecules that fed the description

Sidebar sliders for noise and humidity let the user make the nose's job harder and watch confidence drop.

Full screen-by-screen detail is in `APP_FLOW.md`.

## 6. Functional requirements

| ID | Requirement |
|---|---|
| F1 | `smell_model.py` downloads the OpenPOM CSV if missing, featurizes SMILES, trains a multi-label model, reports macro ROC-AUC on a held-out set, saves model and metadata. |
| F2 | `samples.json` holds six samples, each with a list of molecules (name, SMILES, relative weight). All SMILES must parse in RDKit. |
| F3 | `sensor.py` simulates 8 sensors whose responses depend on molecular properties (size, polarity, oxygen content, sulfur content, nitrogen content, aromaticity), with noise, humidity effect, and baseline drift. |
| F4 | `train_nose.py` generates 200 simulated sessions per sample, trains a Random Forest, reports accuracy on held-out sessions, saves model and metrics. |
| F5 | `pipeline.py` exposes `sniff(sample_name, noise, humidity, seed)` that returns sensor readings, nose prediction, and smell description in one dict. |
| F6 | `app.py` is the Streamlit UI described above. It must fail with a clear message if models are not trained yet. |
| F7 | `test_app.py` runs the pipeline for all six samples and asserts sanity conditions. Exit code 0 on pass. |
| F8 | `requirements.txt` pins versions that install on Python 3.10 to 3.12 without a compiler. |

## 7. Success criteria

| Metric | Target | Notes |
|---|---|---|
| Smell model macro ROC-AUC | at least 0.78 on held-out molecules | The reference write-up reports 0.83. A fingerprint model may land a little lower. Report the measured number, do not edit it. |
| Nose accuracy, default noise | at least 95% on held-out sessions | Same simulator as training, so this is optimistic by design. |
| Nose accuracy, high noise slider | reported, no target | Shows degradation honestly. |
| `smell_model.py` runtime | under 3 minutes on a laptop, first run | Includes download. |
| `train_nose.py` runtime | under 60 seconds | |
| `test_app.py` | passes | |
| Cold start to app in browser | five commands | |

## 8. Honesty requirements

These matter as much as the features.

- The README and the app footer both say the nose is simulated.
- The README explains that the nose accuracy is high because train and test come from one simulator.
- The app does not present descriptor scores as certain. They are model scores for a molecule mix, not measurements of a real smell.
- Sample descriptors are checked against expectation in tests (mint should surface mint or cooling terms, lemon should surface citrus terms). If a check fails, the test prints the actual top descriptors and fails. It does not get loosened silently.

## 9. Risks

| Risk | Handling |
|---|---|
| Dataset URL changes or is blocked | `smell_model.py` accepts `--csv path`. README says where to get the file. |
| RDKit install problems | Use `rdkit` from PyPI wheels. Document fallback of conda. |
| Coffee and wood smoke share molecules (guaiacol, furfural) and may confuse the nose | Distinct weights per sample, plus sensors keyed to properties that differ. If the confusion matrix shows it, report it. |
| Descriptor output looks generic | Rank by lift over the dataset base rate, not raw probability. See `ALGORITHMS.md` section 4. |

## 10. Deliverables

```
smell-ai/
  smell_model.py
  samples.json
  sensor.py
  train_nose.py
  pipeline.py
  app.py
  test_app.py
  requirements.txt
  README.md
  AGENTS.md
  docs/ (PRD, TRD, APP_FLOW, BACKEND_SCHEMA, ALGORITHMS)
```

## 11. Out of scope but documented

README Part 3 describes the BME688 on ESP32 route, the 24 hour burn-in, heater temperature steps (200, 250, 300, 350 C), at least three sessions on different days, room-temperature coffee only, and starting with very different smells. README Part 4 describes OpenPOM's GNN (0.89 ROC-AUC in its authors' tests).
