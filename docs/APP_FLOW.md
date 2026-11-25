# App Flow: Smell AI

## 1. First-run flow (terminal)

```
git clone / cd smell-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python smell_model.py        # downloads data, trains, prints macro ROC-AUC
python train_nose.py         # trains nose, prints accuracy and confusion matrix
python test_app.py           # optional check, exits 0 on pass
streamlit run app.py
```

Expected terminal output from `smell_model.py` ends with a line like:

```
macro ROC-AUC (held-out): 0.8xx over NNN descriptors
saved models/smell.joblib
```

Expected from `train_nose.py`:

```
held-out accuracy: 0.9xx
saved models/nose.joblib
```

## 2. App startup

1. `app.py` calls `load_all()` (cached).
2. If any model file is missing, show an error box listing the missing file and the command that creates it. Stop rendering.
3. Otherwise render the main screen.

## 3. Main screen layout

Single page, wide layout.

```
+-----------------------------------------------------------+
| Title: Smell AI                                           |
| Subtitle: simulated electronic nose + molecule smell model|
+-----------------+-----------------------------------------+
| SIDEBAR         | MAIN                                    |
|  Noise slider   |  [Sample dropdown]   [Sniff button]     |
|  Humidity slider|                                         |
|  Seed input     |  Row 1:  Sensor channels (bar chart)    |
|  (optional)     |                                         |
|                 |  Row 2:  Left: Nose guess + confidence  |
|                 |          Right: probabilities, 6 bars   |
|                 |                                         |
|                 |  Row 3:  Smell description              |
|                 |          top 5 descriptors with scores  |
|                 |                                         |
|                 |  Row 4:  Molecules used (table)         |
+-----------------+-----------------------------------------+
| Footer: the nose is simulated. See README Part 3.         |
+-----------------------------------------------------------+
```

## 4. Controls

| Control | Type | Range | Default | Effect |
|---|---|---|---|---|
| Sample | selectbox | six names from `samples.json` | first | Which smell is presented to the nose. |
| Sniff | button | n/a | n/a | Runs `sniff()` once. |
| Noise | slider | 0.00 to 0.30 | 0.03 | Gaussian noise scale on each channel. |
| Humidity | slider | 0.0 to 1.0 | 0.5 | Shifts sensor baseline and gain. |
| Seed | number input | blank or int | blank | Blank means a fresh random reading each press. A number makes the reading repeatable. |

## 5. Sniff interaction

1. User presses **Sniff**.
2. App calls `sniff(ctx, sample, noise, humidity, seed)`.
3. App stores the result in `st.session_state["last"]`.
4. App renders the four rows from that result.

Changing a slider without pressing Sniff does nothing. This avoids a rerun storm and matches the "pick, then sniff" model.

Before the first Sniff the main area shows one line: "Pick a smell and press Sniff." No empty charts.

## 6. Row details

**Row 1, sensor channels.** Bar chart, 8 bars labelled S1 to S8, y-axis is the reading relative to clean air (1.0 means no change). A thin note under it: "Readings are simulated."

**Row 2, nose guess.**
- Left: large text with the predicted sample name and confidence as a percentage.
- If the prediction differs from the sample picked, show a plain line: "Nose guessed X. You presented Y." No celebration, no alarm styling beyond `st.warning`.
- If top confidence is below 0.5, show "Low confidence" in place of a confident-sounding label.
- Right: bar chart of probabilities for all six samples.

**Row 3, smell description.**
- Heading uses the nose's predicted sample, not the presented one.
- Five descriptors, each with a score from 0 to 1 and a small horizontal bar.
- A one-line caption: "Scores come from the molecule model, averaged over this sample's aroma molecules."

**Row 4, molecules.** Table with columns: name, SMILES, weight. Collapsed in an expander titled "Molecules behind this description".

## 7. Edge cases

| Case | Behavior |
|---|---|
| Model files missing | Error box with the commands. App stops. |
| Nose wrong | Show the mismatch line, show the description for the guessed sample. |
| Noise at 0.30 | Allowed. Accuracy will drop. That is the point of the slider. |
| Sniff pressed repeatedly with blank seed | New reading each time, results may vary. |
| Sniff pressed repeatedly with fixed seed | Identical output each time. |

## 8. Optional stretch screen (build only after everything else passes)

A second tab, "Single molecule". Text input for a SMILES string. On submit: validate with RDKit, show the top 8 descriptors from the smell model. Invalid SMILES shows `st.error("Could not parse that SMILES.")`. Skip this if time is tight. Nothing else depends on it.

## 9. Headless flow (`test_app.py`)

```
load_all() -> for each sample: sniff x20 -> check nose accuracy, keys, descriptor sanity -> print table -> exit 0/1
```

Output is a table: sample, nose accuracy over 20 sniffs, top 5 descriptors, PASS or FAIL.
