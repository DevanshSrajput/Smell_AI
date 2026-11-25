# Algorithms: Smell AI

All math here is small enough to implement directly. Where a choice is arbitrary, the document says so and gives the default.

## 1. Smell model (molecule to descriptors)

### 1.1 Featurization

For each SMILES:

1. Parse with RDKit `Chem.MolFromSmiles`. If it returns `None`, drop the row and count it.
2. Compute a Morgan count fingerprint, radius 2, 2048 bits, with `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)` and `GetCountFingerprintAsNumPy`.
3. Stack into `X` of shape `(n, 2048)`, dtype float32. Optionally apply `log1p` to the counts.

### 1.2 Model

Multi-label classification, 138 outputs. Use one `RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced_subsample")` fitted directly on the 2D label matrix `Y`. scikit-learn's forest supports multi-output natively and `predict_proba` returns a list of 138 arrays, one per output. Take column 1 of each.

Keep the wrapper tiny:

```python
def predict_proba(model, X):
    return np.column_stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(X)) for p in model.predict_proba(X)])
```

The guard handles a descriptor that has only one class in training.

If training exceeds the time budget, reduce `n_estimators` to 150 first, then set `max_depth=30`. Do not switch to a deep model for v1.

Alternative if the forest underperforms: `MLPClassifier(hidden_layer_sizes=(512,), early_stopping=True)`, which is also natively multi-label in scikit-learn. Try it only if macro AUC is under 0.75, and keep whichever is better on the validation split.

### 1.3 Evaluation

1. Random split, 80/20, seed 42. (A scaffold split is stricter and is a good README note, but is not required.)
2. For each descriptor with at least one positive and one negative in the test set, compute `roc_auc_score`.
3. Macro ROC-AUC is the mean over those descriptors.
4. Save per-descriptor AUC in the metrics file.

Report exactly what is measured. The reference write-up got 0.83. A fingerprint forest can land around 0.78 to 0.85. Do not tune against the test set; if tuning is needed, carve a validation split from train.

### 1.4 Final fit

After evaluation, retrain on all molecules for the saved model? Default: no. Save the model trained on the 80% split so the saved model matches the reported metric. Note this in the metrics file.

## 2. Sensor simulator

### 2.1 Molecule property vector

For each molecule, compute 8 properties with RDKit:

| Index | Name | How |
|---|---|---|
| 0 | `mw` | `Descriptors.MolWt` |
| 1 | `logp` | `Crippen.MolLogP` (oil-solubility proxy) |
| 2 | `o_frac` | oxygen atoms divided by heavy atoms |
| 3 | `s_count` | sulfur atom count |
| 4 | `n_count` | nitrogen atom count |
| 5 | `aromatic_rings` | `rdMolDescriptors.CalcNumAromaticRings` |
| 6 | `tpsa` | `rdMolDescriptors.CalcTPSA` (polarity) |
| 7 | `hbd` | `rdMolDescriptors.CalcNumHBD` |

Standardize: `z = (f - mean) / std`, with mean and std stored in `sensor_bank.json`.

### 2.2 Sensor bank

Seed 7. For each sensor `i` in 0..7:

- `w_i` = a vector of 8 values drawn from Normal(0, 1), then sparsified: zero out a random 4 of 8 entries so each sensor keys on a few properties, as real metal-oxide sensors do.
- `b_i` ~ Normal(0, 0.5)
- `gain_i` ~ Uniform(0.4, 1.0)
- `humidity_coeff_i` ~ Uniform(0.05, 0.25)
- `drift_coeff_i` ~ Uniform(0.00, 0.03)

Check after generation: the 8 weight vectors must not be near-duplicates. If any pair has cosine similarity above 0.9, redraw with seed 7 plus an increment until none do. Record the final seed used.

### 2.3 Volatility

Heavier molecules reach the sensor less. For a molecule with molecular weight `mw`:

```
volatility = exp(-(mw - 100) / 120)    # clipped to [0.2, 1.5]
```

Arbitrary but monotone. It is documented as a modelling choice.

### 2.4 Response

For one presented sample with molecules `m = 1..M`, normalized weights `c_m`, and a per-sniff intensity `k ~ Uniform(0.6, 1.4)` (how strongly the sample is presented):

```
a_im   = sigmoid(w_i . z_m + b_i)                 # affinity of sensor i for molecule m, in (0,1)
signal_i = gain_i * k * sum_m c_m * volatility_m * a_im
```

Saturate the signal so loud smells flatten out like real sensors:

```
sat_i = 1 - exp(-3 * signal_i)
```

Environment terms:

```
h      = humidity in [0, 1]
env_i  = humidity_coeff_i * (h - 0.5)              # shifts the reading with humidity
drift  = drift_coeff_i * u,  u ~ Uniform(-1, 1)    # per-sniff baseline wobble
noise  = Normal(0, sigma)                          # sigma is the noise slider
```

Final reading, expressed relative to clean air (1.0 means no change):

```
R_i = 1 + sat_i + env_i + drift_i + noise_i
```

Return the 8 values `R_0..R_7`. Clip to a minimum of 0.5 so noise cannot make nonsense negative readings.

Because the same property vector feeds both the sensors and the physical story, the nose sees real structure in each sample, not labels.

## 3. Nose classifier

### 3.1 Training data

For each of six samples, simulate 200 sessions. Per session, draw independently:

- `noise sigma ~ Uniform(0.01, 0.06)`
- `humidity ~ Uniform(0.2, 0.8)`
- intensity `k`, drift, and noise from the simulator

Group by session index, not by row, since each session is one row. Split 80/20 stratified by sample, seed 1. The test sessions are never used in training.

### 3.2 Model

`RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1)` on the raw 8-vector. No scaling needed.

### 3.3 Evaluation

- Held-out accuracy at the default noise range.
- Confusion matrix, saved in the metrics file.
- Degradation curve: generate 300 fresh sessions per sample at fixed sigma in {0.03, 0.10, 0.20, 0.30}, score with the trained model, record accuracy per sigma.

If held-out accuracy is above 99.9%, the simulator may be too easy. Check that the six sample signatures are not trivially separable by one channel. A mild tightening (more sparsity overlap between sensors, or larger drift) is allowed. Record any change in the metrics file.

### 3.3.1 Why the number is optimistic

The test sessions come from the same simulator that made the training sessions. High accuracy shows the pipeline works. It says nothing about real hardware. The README must say this in the results section.

### 3.4 Confidence rule

Confidence is the top class probability from `predict_proba`. The UI labels confidence under 0.5 as "Low confidence".

## 4. Describing a sample in words

Given the predicted sample with molecules `m = 1..M` and normalized weights `c_m`:

1. `P_m` = smell model probabilities for molecule m, a vector of length 138. Precomputed.
2. Sample score: `S = sum_m c_m * P_m`, a vector of length 138.
3. Lift: `L_d = S_d / max(base_rate_d, 0.005)`.
4. Ranking: sort descriptors by `S_d * sqrt(L_d)`. This favors descriptors that are both probable and distinctive, without letting rare descriptors with tiny probabilities jump to the top.
5. Return the top 5 as `{descriptor, score = S_d, lift = L_d}`.

Why not rank by raw score: common descriptors like `sweet` or `fruity` have high base rates and would dominate every sample. The lift term brings out what is characteristic.

If the top 5 look generic after this, try the exponent 1.0 on lift instead of 0.5 and compare for all six samples. Pick one setting for all samples and say which in the README. Do not tune per sample.

## 5. Pipeline orchestration

```
sniff(ctx, sample_id, noise, humidity, seed):
    rng = np.random.default_rng(seed)
    molecules = ctx.samples[sample_id].molecules
    readings  = sensor.simulate(molecules, noise, humidity, rng)       # shape (8,)
    proba     = ctx.nose.predict_proba(readings.reshape(1, -1))[0]
    predicted = ctx.nose_classes[argmax(proba)]
    desc      = describe(ctx, predicted, top_k=5)                      # section 4, uses precomputed P_m
    return assemble_dict(...)
```

`seed=None` gives a fresh `default_rng()`.

## 6. Complexity

Everything is linear in the number of molecules and trivially small at runtime. Training the smell forest dominates total compute.

## 7. Real-sensor swap (reference)

For a BME688 or BME680: at each heater temperature (for example 200, 250, 300, 350 C) read gas resistance, divide by the clean-air reading, and concatenate across temperatures to form the feature vector. One row per heater cycle. Train the same Random Forest. Split by session, not by row, so no session leaks across train and test. Record each smell in its own jar for 30 minutes with clean air between, on at least three different days. Use room-temperature coffee grounds. Hot coffee mostly teaches the sensor to detect steam. Start with very different smells.
