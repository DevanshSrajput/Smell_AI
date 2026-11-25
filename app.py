"""SMELL//OS — Minimal light Y2K Streamlit interface for the Smell AI pipeline."""

from __future__ import annotations

import json
import platform
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import pipeline
import sensor
import smell_model
import train_nose

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
SAMPLES_PATH = ROOT / "samples.json"
LOG_PATH = pipeline.LOG_PATH

st.set_page_config(
    page_title="SMELL//OS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
  --bg: #f4f7fc; --panel: #ffffff; --panel2: #eef3fa;
  --blue: #0066ff; --cyan: #00a8ff; --pink: #ff3399; --mint: #00a878; --amber: #d97706;
  --text: #162033; --muted: #5c6f8a; --line: #d4dfee;
}
.stApp {
  background: linear-gradient(180deg, #f8faff 0%, #eef3fa 100%);
  color: var(--text);
  font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid var(--line);
}
h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: .04em;
  color: var(--text) !important;
}
p, li, label, span, div { color: var(--text); }
a { color: var(--blue); }
code {
  font-family: 'IBM Plex Mono', monospace;
  color: #0052cc; background: #edf3ff;
  border: 1px solid #d5e3ff; border-radius: 6px; padding: 1px 5px;
}
pre {
  background: #f8faff !important; border: 1px solid var(--line) !important;
  border-radius: 10px !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
  background: #ffffff; border: 1px solid var(--line);
  border-radius: 999px; color: var(--muted);
  font-family: 'IBM Plex Mono', monospace; letter-spacing: .04em;
  font-size: .74rem; padding: 6px 14px; margin-bottom: 6px;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(180deg, #ffffff 0%, #e5f0ff 100%);
  color: var(--blue); border-color: var(--blue);
  box-shadow: 0 2px 8px rgba(0, 102, 255, .12);
}
.stButton > button {
  background: linear-gradient(180deg, #ffffff 0%, #e9f1ff 100%);
  color: var(--blue); border: 1px solid #99c2ff; border-radius: 999px;
  font-family: 'IBM Plex Mono', monospace; letter-spacing: .06em;
  font-size: .78rem; text-transform: uppercase;
  box-shadow: 0 2px 6px rgba(0, 102, 255, .08);
  transition: all .15s ease;
}
.stButton > button:hover {
  background: var(--blue); color: #ffffff; border-color: var(--blue);
  box-shadow: 0 4px 12px rgba(0, 102, 255, .22);
}
[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 12px; padding: 12px 14px;
  box-shadow: 0 2px 10px rgba(22, 32, 51, .03);
}
[data-testid="stMetricLabel"] {
  color: var(--muted) !important; text-transform: uppercase;
  font-family: 'IBM Plex Mono', monospace;
  letter-spacing: .08em; font-size: .68rem;
}
[data-testid="stMetricValue"] {
  color: var(--blue) !important; font-family: 'Space Grotesk', sans-serif;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--line); border-radius: 10px; background: #ffffff;
}
[data-testid="stExpander"] {
  background: #ffffff; border: 1px solid var(--line); border-radius: 12px;
}
[data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="base-input"] {
  background: #ffffff !important; border-color: var(--line) !important; border-radius: 8px !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--blue); box-shadow: 0 0 0 3px rgba(0, 102, 255, .15);
}
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--blue), var(--cyan));
}
.stAlert { background: #fff5fa; border: 1px solid #ffcce6; border-radius: 10px; }
hr { border-color: var(--line); }
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.hero {
  border: 1px solid var(--line); border-radius: 16px;
  padding: 20px 24px; margin-bottom: 16px;
  background: linear-gradient(135deg, #ffffff 0%, #eef5ff 60%, #fdf2f8 100%);
  box-shadow: 0 4px 20px rgba(22, 32, 51, .04);
}
.hero-kicker {
  font-family: 'IBM Plex Mono', monospace; color: var(--muted);
  font-size: .70rem; letter-spacing: .16em;
}
.hero-title {
  font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.8rem, 4vw, 2.8rem);
  font-weight: 700; letter-spacing: .04em; line-height: 1.1;
  color: var(--text); margin: 4px 0;
}
.hero-title span { color: var(--blue); }
.hero-sub {
  font-family: 'IBM Plex Mono', monospace; color: var(--muted);
  font-size: .74rem; letter-spacing: .08em; margin-bottom: 10px;
}
.badge {
  display: inline-block; padding: 3px 10px; margin: 2px 6px 2px 0;
  border-radius: 999px; border: 1px solid #b8d4ff; color: var(--blue);
  background: #f0f6ff; font-family: 'IBM Plex Mono', monospace;
  font-size: .64rem; letter-spacing: .08em;
}
.badge-magenta { border-color: #ffb8dc; color: var(--pink); background: #fff0f8; }
.badge-lime { border-color: #a7e8d2; color: var(--mint); background: #edfcf6; }
.badge-amber { border-color: #fcd34d; color: var(--amber); background: #fffbeb; }
.section-tag {
  font-family: 'IBM Plex Mono', monospace;
  color: var(--blue); font-size: .72rem; letter-spacing: .14em;
  text-transform: uppercase; margin: 12px 0 6px;
}
.terminal {
  background: #ffffff; border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 14px; color: var(--text);
  font-family: 'IBM Plex Mono', monospace; font-size: .82rem;
}
.terminal .prompt { color: var(--blue); font-weight: 500; }
.cursor {
  display: inline-block; width: 7px; height: 13px; background: var(--blue);
  animation: blink 1s steps(2) infinite; vertical-align: middle;
}
@keyframes blink { 0%,49% { opacity: 1; } 50%,100% { opacity: 0; } }
.sidebar-brand {
  font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700;
  color: var(--text); letter-spacing: .04em;
}
.sidebar-brand span { color: var(--blue); }
.sidebar-sub {
  font-family: 'IBM Plex Mono', monospace; color: var(--muted);
  font-size: .64rem; letter-spacing: .12em;
}
.footer {
  border-top: 1px solid var(--line); margin-top: 24px; padding-top: 12px;
  color: var(--muted); font-family: 'IBM Plex Mono', monospace;
  font-size: .70rem; letter-spacing: .04em;
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


@st.cache_resource
def _load_context() -> pipeline.Context:
    return pipeline.load_all()


@st.cache_data
def _load_metrics() -> dict:
    metrics = {}
    for name in ("smell_metrics", "nose_metrics", "gnn_metrics"):
        path = MODELS_DIR / f"{name}.json"
        if path.exists():
            metrics[name] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def _log_count() -> int:
    if not LOG_PATH.exists():
        return 0
    with LOG_PATH.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _section(tag: str) -> None:
    st.markdown(f'<div class="section-tag">{tag}</div>', unsafe_allow_html=True)


def _terminal(text: str) -> None:
    st.markdown(
        f'<div class="terminal"><span class="prompt">&gt;</span> {text}</div>',
        unsafe_allow_html=True,
    )


def _description_bars(items: list[dict]) -> None:
    for item in items:
        st.markdown(
            f"**{item['descriptor']}** · `score {item['score']:.3f}` · "
            f"`lift {item['lift']:.1f}×`"
        )
        st.progress(float(min(max(item["score"], 0.0), 1.0)))


def _heatmap(matrix: pd.DataFrame):
    vmax = max(int(matrix.values.max()), 1)

    def color(value: float) -> str:
        ratio = float(value) / vmax
        intensity = 0.06 + 0.65 * ratio
        ink = "#ffffff" if ratio > 0.6 else "#162033"
        return f"background-color: rgba(0,102,255,{intensity:.2f}); color: {ink};"

    styler = matrix.style
    if hasattr(styler, "map"):
        return styler.map(color).format("{:.0f}")
    return styler.applymap(color).format("{:.0f}")  # pragma: no cover


def _table(data, **kwargs):
    """Render a dataframe with the current width API, falling back for older Streamlit."""
    kwargs.pop("width", None)
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def _hero(loaded: bool = True, backend: str = "rf") -> None:
    model_badge = (
        '<span class="badge badge-lime">MODELS: LOADED</span>'
        if loaded
        else '<span class="badge badge-amber">MODELS: MISSING</span>'
    )
    system_badge = (
        '<span class="badge badge-lime">● SYSTEM ONLINE</span>'
        if loaded
        else '<span class="badge badge-amber">● SYSTEM BOOT</span>'
    )
    arch_badge = (
        '<span class="badge">SMELL: OPENPOM MPNN (GNN)</span>'
        if backend == "gnn"
        else '<span class="badge">SMELL: RANDOM FOREST (CPU)</span>'
    )
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">// OLFACTORY INFERENCE ENGINE</div>
          <div class="hero-title">SMELL<span>//</span>OS</div>
          <div class="hero-sub">138-DIMENSIONAL MOLECULE OLFACTION · 8-CHANNEL SIMULATED NOSE</div>
          <div class="hero-status">
            {system_badge}
            {arch_badge}
            <span class="badge badge-magenta">NOSE: SIMULATED</span>
            {model_badge}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _bootstrap_screen() -> None:
    _hero(loaded=False)
    missing = pipeline.missing_model_files()
    st.error("Model artifacts are missing. This runtime has not been initialized.")
    st.markdown("Missing artifacts:")
    for rel in missing:
        st.markdown(f"- `{rel}` — `{pipeline.REQUIRED_FILES[rel]}`")

    st.markdown("### Initialize in this runtime")
    st.write(
        "The app can download the OpenPOM CSV and train both models here. "
        "First initialization takes roughly 1–3 minutes on a laptop CPU."
    )
    if st.button("INITIALIZE MODELS", type="primary", key="bootstrap_btn"):
        smell_missing = any(
            rel in missing
            for rel in (
                "models/smell.joblib",
                "models/descriptors.json",
                "models/smell_metrics.json",
            )
        )
        nose_missing = any(
            rel in missing
            for rel in (
                "models/sensor_bank.json",
                "models/nose.joblib",
                "models/nose_metrics.json",
            )
        )
        with st.status("Initializing model stack...", expanded=True) as status:
            try:
                if smell_missing:
                    st.write("> training smell model (downloads the OpenPOM CSV)")
                    smell_model.train()
                if nose_missing:
                    st.write("> training simulated nose")
                    train_nose.train()
                status.update(label="Model stack ready", state="complete")
            except SystemExit:
                status.update(label="Initialization failed", state="error")
                st.error(
                    "Dataset download failed. Check outbound network access, or train "
                    "locally and mount the models/ directory."
                )
                st.code("python smell_model.py\npython train_nose.py")
                return
            except Exception as exc:
                status.update(label="Initialization failed", state="error")
                st.exception(exc)
                return
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(
        "Alternative: run `python smell_model.py` and `python train_nose.py` locally, "
        "then deploy with the models present."
    )


def _sidebar(ctx: pipeline.Context, metrics: dict) -> tuple[str, float, float, int | None, str]:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">SMELL<span>//</span>OS</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">OLFACTORY INFERENCE CONSOLE</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("#### // SESSION CONTROL")
        backend = "rf"
        if ctx.gnn_model is not None:
            choice = st.radio(
                "Smell Model",
                ["OpenPOM MPNN (GNN)", "RandomForest (CPU)"],
                index=0,
                key="ctl_backend",
            )
            backend = "gnn" if "MPNN" in choice else "rf"
        picked = st.selectbox(
            "Sample",
            list(ctx.samples),
            format_func=lambda sample_id: ctx.samples[sample_id].label,
            key="ctl_sample",
        )
        noise = st.slider("Noise σ", 0.00, 0.30, 0.03, 0.01, key="ctl_noise")
        humidity = st.slider("Humidity", 0.0, 1.0, 0.5, 0.05, key="ctl_humidity")
        seed = st.number_input(
            "Seed (blank = fresh)",
            min_value=0,
            value=None,
            step=1,
            placeholder="blank",
            key="ctl_seed",
        )
        st.caption("Fixed seed = repeatable sniff. Blank = fresh RNG each press.")
        st.divider()
        st.markdown("#### // SYSTEM STATUS")

        smell = metrics.get("smell_metrics", {})
        gnn = metrics.get("gnn_metrics", {})
        nose = metrics.get("nose_metrics", {})
        active_auc = gnn.get("random_split_macro_roc_auc", 0.0) if backend == "gnn" else smell.get("macro_roc_auc", 0.0)
        left, right = st.columns(2)
        left.metric("GNN AUC" if backend == "gnn" else "RF AUC", f"{active_auc:.3f}")
        right.metric("NOSE ACC", f"{nose.get('accuracy_default_noise', 0.0):.1%}")
        st.caption("Measured JSON metrics. NOSE: SIMULATED.")

        with st.expander("SYSTEM"):
            st.write(f"Python `{platform.python_version()}`")
            ready = len(pipeline.REQUIRED_FILES) - len(pipeline.missing_model_files())
            st.write(f"Models `{ready}/{len(pipeline.REQUIRED_FILES)}` (GNN: `{'ready' if ctx.gnn_model else 'absent'}`)")
            st.write(f"Log lines `{_log_count()}`")
            st.write(f"Sensor seed `{ctx.bank.seed}`")
            if st.button("RESET SESSION", key="reset_btn"):
                for key in (
                    "last_sniff",
                    "batch_df",
                    "molecule_result",
                    "sweep_df",
                    "sweep_meta",
                ):
                    st.session_state.pop(key, None)
                st.rerun()
        st.caption("SIMULATED NOSE — NOT HARDWARE")
    return picked, noise, humidity, seed, backend


def _sniff_lab(
    ctx: pipeline.Context, picked: str, noise: float, humidity: float, seed: int | None, backend: str = "rf"
) -> None:
    _section("01 // SINGLE SNIFF")
    st.write(
        "Pick a smell, set the environment, and initiate a sniff. The nose reads eight "
        "simulated channels; the selected smell model describes the molecules behind its guess."
    )
    if st.button("INITIATE SNIFF", type="primary", key="sniff_btn"):
        try:
            st.session_state["last_sniff"] = pipeline.sniff(
                ctx, picked, noise=noise, humidity=humidity, seed=seed, backend=backend
            )
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.get("last_sniff")
    if result is None:
        _terminal('AWAITING SAMPLE ... <span class="cursor"></span>')
        return

    presented_label = ctx.samples[result["presented"]].label
    predicted_label = ctx.samples[result["nose"]["predicted"]].label
    confidence = result["nose"]["confidence"]
    seed_label = "fresh" if result["conditions"]["seed"] is None else result["conditions"]["seed"]
    used_backend = "OpenPOM MPNN (GNN)" if result.get("smell_backend") == "gnn" else "RandomForest (CPU)"
    st.caption(
        f"noise σ={result['conditions']['noise']:.2f} · "
        f"humidity={result['conditions']['humidity']:.2f} · seed={seed_label} · smell={used_backend}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PRESENTED", presented_label)
    c2.metric("NOSE GUESS", predicted_label)
    c3.metric("CONFIDENCE", f"{confidence:.1%}")
    c4.metric("VERDICT", "MATCH" if result["nose"]["correct"] else "MISMATCH")

    if confidence < 0.5:
        st.warning("Low confidence: the top class is below 50%.")
    if not result["nose"]["correct"]:
        st.warning(
            f"Nose guessed {predicted_label}. You presented {presented_label}. "
            "The description below follows the nose's guess."
        )

    st.subheader("Sensor array")
    sensor_df = pd.DataFrame(
        {"R/R₀": result["sensors"]["values"]}, index=result["sensors"]["names"]
    )
    st.bar_chart(sensor_df, height=280)
    st.caption("R/R₀ relative to clean air. 1.0 means no change. Simulated channels, not hardware.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Nose output")
        prob_df = pd.DataFrame(
            {
                "probability": [
                    result["nose"]["probabilities"][sample_id]
                    for sample_id in ctx.nose_classes
                ]
            },
            index=[ctx.samples[sample_id].label for sample_id in ctx.nose_classes],
        )
        st.bar_chart(prob_df, height=280)
    with right:
        st.subheader("Top descriptors")
        _description_bars(result["description"])

    with st.expander("MOLECULES BEHIND THIS DESCRIPTION"):
        _table(
            pd.DataFrame(
                [
                    {
                        "name": molecule["name"],
                        "SMILES": molecule["smiles"],
                        "weight": molecule["weight"],
                    }
                    for molecule in result["molecules"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    dl, trace = st.columns([1, 1])
    with dl:
        st.download_button(
            "DOWNLOAD SNIFF JSON",
            json.dumps(result, indent=2),
            file_name=f"sniff_{result['presented']}.json",
            mime="application/json",
            key="dl_sniff",
        )
    with trace:
        with st.expander("INFERENCE TRACE"):
            st.json(result)

def _batch_arena(ctx: pipeline.Context, noise: float, humidity: float) -> None:
    _section("02 // BATCH ARENA")
    st.write(
        "Run repeated sniffs across all six samples under the current conditions. "
        "This is the same evaluation style as the headless test, with a live confusion matrix."
    )
    repeats = st.slider("Sniffs per sample", 5, 30, 10, 1, key="batch_repeats")
    st.caption(f"Conditions: noise σ={noise:.2f} · humidity={humidity:.2f}")
    if st.button("RUN BATCH", type="primary", key="batch_run"):
        rows = []
        total = len(ctx.samples) * repeats
        progress = st.progress(0.0, text="Running batch...")
        for i, sample_id in enumerate(ctx.samples):
            for j in range(repeats):
                result = pipeline.sniff(
                    ctx,
                    sample_id,
                    noise=noise,
                    humidity=humidity,
                    seed=30_000 + i * 100 + j,
                    log=False,
                )
                rows.append(
                    {
                        "presented": sample_id,
                        "predicted": result["nose"]["predicted"],
                        "correct": bool(result["nose"]["correct"]),
                        "confidence": float(result["nose"]["confidence"]),
                        "top_descriptor": result["description"][0]["descriptor"],
                        "noise": float(noise),
                        "humidity": float(humidity),
                    }
                )
                progress.progress(len(rows) / total, text=f"{len(rows)}/{total} sniffs")
        progress.empty()
        st.session_state["batch_df"] = pd.DataFrame(rows)

    df = st.session_state.get("batch_df")
    if df is None:
        _terminal('NO BATCH IN MEMORY ... <span class="cursor"></span>')
        return

    accuracy = float(df["correct"].mean())
    mean_confidence = float(df["confidence"].mean())
    c1, c2, c3 = st.columns(3)
    c1.metric("ACCURACY", f"{accuracy:.1%}")
    c2.metric("MEAN CONFIDENCE", f"{mean_confidence:.1%}")
    c3.metric("SNIFFS", f"{len(df)}")
    st.caption(
        f"Batch conditions: noise σ={df['noise'].iloc[0]:.2f} · "
        f"humidity={df['humidity'].iloc[0]:.2f}"
    )

    per_sample = df.groupby("presented", as_index=False).agg(
        accuracy=("correct", "mean"),
        confidence=("confidence", "mean"),
        n=("correct", "size"),
    )
    st.subheader("Per-sample accuracy")
    _table(per_sample, hide_index=True, width="stretch")
    st.bar_chart(per_sample.set_index("presented")["accuracy"], height=240)

    st.subheader("Confusion matrix")
    st.caption("Rows = presented, columns = predicted.")
    classes = list(ctx.samples)
    matrix = pd.crosstab(df["presented"], df["predicted"]).reindex(
        index=classes, columns=classes, fill_value=0
    )
    _table(_heatmap(matrix), width="stretch")

    st.subheader("Top descriptor frequency")
    frequency = df["top_descriptor"].value_counts().head(10)
    st.bar_chart(frequency, height=240)
    st.download_button(
        "DOWNLOAD BATCH CSV",
        df.to_csv(index=False),
        file_name="batch_sniffs.csv",
        mime="text/csv",
        key="dl_batch",
    )


def _molecule_lab(ctx: pipeline.Context, backend: str = "rf") -> None:
    _section("03 // MOLECULE LAB")
    st.write(
        "Analyze one SMILES with the smell model, inspect the eight simulator "
        "properties, and compare RandomForest vs. OpenPOM MPNN predictions."
    )
    smiles = st.text_input("SMILES", value="CC(C)C1CCC(C)CC1O", key="mol_smiles")
    if st.button("ANALYZE MOLECULE", type="primary", key="mol_run"):
        try:
            descriptors_rf = pipeline.molecule_descriptors(ctx, smiles, top_k=12, backend="rf")
            descriptors_gnn = (
                pipeline.molecule_descriptors(ctx, smiles, top_k=12, backend="gnn")
                if ctx.gnn_model is not None
                else None
            )
            features = sensor.molecule_features(smiles)
            reading = sensor.simulate(
                [{"smiles": smiles, "weight": 1.0}],
                0.0,
                0.5,
                np.random.default_rng(0),
            )
            probabilities = ctx.nose.predict_proba(reading.reshape(1, -1))[0]
            st.session_state["molecule_result"] = {
                "smiles": smiles,
                "descriptors": descriptors_gnn if (backend == "gnn" and descriptors_gnn) else descriptors_rf,
                "descriptors_rf": descriptors_rf,
                "descriptors_gnn": descriptors_gnn,
                "features": [float(value) for value in features],
                "reading": [float(value) for value in reading],
                "probabilities": {
                    name: float(probability)
                    for name, probability in zip(ctx.nose_classes, probabilities)
                },
            }
        except ValueError:
            st.session_state.pop("molecule_result", None)
            st.error("Could not parse that SMILES.")

    result = st.session_state.get("molecule_result")
    if result is None:
        _terminal('AWAITING SMILES ... <span class="cursor"></span>')
        return

    st.caption(f"Analyzed `{result['smiles']}`")
    if result.get("descriptors_gnn"):
        c_gnn, c_rf = st.columns([1, 1])
        with c_gnn:
            st.subheader("OpenPOM MPNN (GNN)")
            _description_bars(result["descriptors_gnn"][:8])
        with c_rf:
            st.subheader("RandomForest (CPU)")
            _description_bars(result["descriptors_rf"][:8])
        st.subheader("Sensor properties")
        features = np.asarray(result["features"])
        feature_df = pd.DataFrame(
            {
                "feature": sensor.FEATURE_NAMES,
                "value": features,
                "z-score": (features - ctx.bank.feature_mean) / ctx.bank.feature_std,
            }
        )
        _table(feature_df, hide_index=True, width="stretch")
    else:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Descriptor profile")
            _description_bars(result["descriptors"])
        with right:
            st.subheader("Sensor properties")
            features = np.asarray(result["features"])
            feature_df = pd.DataFrame(
                {
                    "feature": sensor.FEATURE_NAMES,
                    "value": features,
                    "z-score": (features - ctx.bank.feature_mean) / ctx.bank.feature_std,
                }
            )
            _table(feature_df, hide_index=True, width="stretch")

    st.subheader("Nose resemblance")
    prob_df = pd.DataFrame(
        {
            "probability": [
                result["probabilities"][sample_id] for sample_id in ctx.nose_classes
            ]
        },
        index=[ctx.samples[sample_id].label for sample_id in ctx.nose_classes],
    )
    st.bar_chart(prob_df, height=260)
    st.caption(
        "Resemblance only: the nose was trained on six mixtures, not single molecules. "
        "This is not an identification."
    )
    st.download_button(
        "DOWNLOAD MOLECULE JSON",
        json.dumps(result, indent=2),
        file_name="molecule.json",
        mime="application/json",
        key="dl_mol",
    )


def _sensor_sweep(ctx: pipeline.Context, noise: float, humidity: float) -> None:
    _section("04 // SENSOR SWEEP")
    st.write(
        "Sweep one environmental axis and watch the nose's accuracy and confidence degrade."
    )
    mode = st.radio("Axis", ["Noise σ", "Humidity"], horizontal=True, key="sweep_mode")
    levels = st.slider("Levels", 5, 15, 9, 1, key="sweep_levels")
    repeats = st.slider("Sniffs per sample per level", 2, 10, 4, 1, key="sweep_repeats")
    if mode == "Noise σ":
        axis_values = np.linspace(0.0, 0.30, levels)
        st.caption(f"Humidity fixed at {humidity:.2f}; noise swept 0.00 → 0.30.")
    else:
        axis_values = np.linspace(0.0, 1.0, levels)
        st.caption(f"Noise fixed at {noise:.2f}; humidity swept 0.00 → 1.00.")
    total = levels * len(ctx.samples) * repeats
    st.caption(f"{total} sniffs ≈ {total * 0.03:.1f} s at ~30 ms/sniff.")

    if st.button("RUN SWEEP", type="primary", key="sweep_run"):
        rows = []
        progress = st.progress(0.0, text="Sweeping...")
        for level_index, level in enumerate(axis_values):
            for i, sample_id in enumerate(ctx.samples):
                for j in range(repeats):
                    sweep_noise = float(level) if mode == "Noise σ" else float(noise)
                    sweep_humidity = float(humidity) if mode == "Noise σ" else float(level)
                    result = pipeline.sniff(
                        ctx,
                        sample_id,
                        noise=sweep_noise,
                        humidity=sweep_humidity,
                        seed=40_000 + level_index * 1000 + i * 100 + j,
                        log=False,
                    )
                    rows.append(
                        {
                            "level": float(level),
                            "sample": sample_id,
                            "correct": bool(result["nose"]["correct"]),
                            "confidence": float(result["nose"]["confidence"]),
                        }
                    )
                    progress.progress(len(rows) / total, text=f"{len(rows)}/{total} sniffs")
        progress.empty()
        st.session_state["sweep_df"] = pd.DataFrame(rows)
        st.session_state["sweep_meta"] = {
            "mode": mode,
            "noise": float(noise),
            "humidity": float(humidity),
        }

    df = st.session_state.get("sweep_df")
    if df is None:
        _terminal('NO SWEEP IN MEMORY ... <span class="cursor"></span>')
        return

    meta = st.session_state.get("sweep_meta", {})
    st.caption(
        f"Last sweep: {meta.get('mode', '?')} · "
        f"fixed noise {meta.get('noise', 0.0):.2f} · "
        f"fixed humidity {meta.get('humidity', 0.0):.2f}"
    )
    summary = df.groupby("level", as_index=False).agg(
        accuracy=("correct", "mean"), confidence=("confidence", "mean")
    )
    st.subheader("Aggregate curve")
    st.line_chart(summary.set_index("level")[["accuracy", "confidence"]], height=300)
    _table(summary, hide_index=True, width="stretch")

    st.subheader("Per-sample accuracy")
    pivot = df.groupby(["level", "sample"])["correct"].mean().unstack()
    st.line_chart(pivot, height=300)
    st.download_button(
        "DOWNLOAD SWEEP CSV",
        df.to_csv(index=False),
        file_name="sensor_sweep.csv",
        mime="text/csv",
        key="dl_sweep",
    )

def _model_metrics(metrics: dict) -> None:
    _section("05 // MODEL METRICS")
    smell = metrics.get("smell_metrics")
    nose = metrics.get("nose_metrics")
    gnn = metrics.get("gnn_metrics")
    if not smell or not nose:
        st.info("Metrics JSON not found. Run the training commands.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RF MACRO AUC", f"{smell['macro_roc_auc']:.3f}")
    if gnn:
        c2.metric(
            "GNN MACRO AUC",
            f"{gnn['random_split_macro_roc_auc']:.3f}",
            delta=f"+{gnn['random_split_macro_roc_auc'] - smell['macro_roc_auc']:.3f} vs RF",
        )
    else:
        c2.metric("DESCRIPTORS SCORED", f"{smell['n_descriptors_scored']}/{smell['n_descriptors']}")
    c3.metric("NOSE ACCURACY", f"{nose['accuracy_default_noise']:.1%}")
    c4.metric("HELD-OUT SESSIONS", f"{nose['n_test']}")
    st.caption("All values are read from models/*.json; nothing is hardcoded.")

    tab_names = ["RF vs. GNN BENCHMARK", "SMELL MODEL (RF)", "NOSE MODEL"] if gnn else ["SMELL MODEL", "NOSE MODEL"]
    model_tabs = st.tabs(tab_names)
    if gnn:
        tab_bench, tab_smell, tab_nose = model_tabs
        with tab_bench:
            b1, b2, b3 = st.columns(3)
            b1.metric("GNN Random Split AUC", f"{gnn['random_split_macro_roc_auc']:.4f}")
            b2.metric(
                "GNN Scaffold Split AUC",
                f"{gnn['scaffold_split_macro_roc_auc']:.4f}",
                delta=f"+{gnn['scaffold_split_macro_roc_auc'] - gnn['rf_scaffold_split_macro_roc_auc']:.4f} vs RF",
            )
            b3.metric("RF Scaffold Split AUC", f"{gnn['rf_scaffold_split_macro_roc_auc']:.4f}")

            comp = pd.DataFrame(
                [
                    {
                        "Model": "RandomForest (CPU Baseline)",
                        "Input": "2048-bit Morgan Count FP",
                        "Hardware": "Laptop CPU",
                        "Random 80/20 AUC": round(smell["macro_roc_auc"], 4),
                        "Scaffold 80/20 AUC": round(gnn["rf_scaffold_split_macro_roc_auc"], 4),
                        "Train Time (s)": smell["train_seconds"],
                    },
                    {
                        "Model": "OpenPOM MPNN (GNN)",
                        "Input": "Atom/Bond Graph (3-step MPNN)",
                        "Hardware": gnn.get("device", "GPU"),
                        "Random 80/20 AUC": round(gnn["random_split_macro_roc_auc"], 4),
                        "Scaffold 80/20 AUC": round(gnn["scaffold_split_macro_roc_auc"], 4),
                        "Train Time (s)": gnn["train_seconds"],
                    },
                ]
            )
            _table(comp, hide_index=True, width="stretch")
            if gnn.get("history"):
                hist_df = pd.DataFrame(gnn["history"]).set_index("epoch")
                st.subheader("GNN training curve (Random 80/20)")
                st.line_chart(hist_df[["val_auc", "loss"]], height=240)
    else:
        tab_smell, tab_nose = model_tabs
    with tab_smell:
        left, right = st.columns(2)
        left.write(f"**Featurizer** `{smell['featurizer']}`")
        left.write(f"**Model** `{smell['model']}`")
        left.write(f"**Split** `{smell['split']}`")
        left.write(f"**Fit seconds** `{smell['train_seconds']}`")
        right.write(f"**Molecules total** `{smell['n_molecules_total']}`")
        right.write(f"**Molecules used** `{smell['n_molecules_used']}`")
        right.write(f"**Dropped unparsable** `{smell['n_dropped_unparsable']}`")
        right.write(f"**Fitted on** `{smell.get('fitted_on', 'train_split')}`")

        per_descriptor = pd.DataFrame(
            sorted(
                smell["per_descriptor_auc"].items(),
                key=lambda item: item[1],
                reverse=True,
            ),
            columns=["descriptor", "auc"],
        )
        query = st.text_input(
            "Filter descriptors", key="auc_filter", placeholder="e.g. mint"
        )
        if query:
            per_descriptor = per_descriptor[
                per_descriptor["descriptor"].str.contains(query, case=False, regex=False)
            ]
        _table(per_descriptor, hide_index=True, width="stretch")
        if not per_descriptor.empty:
            st.bar_chart(
                per_descriptor.head(15).set_index("descriptor")["auc"], height=260
            )
        else:
            st.info("No descriptors match that filter.")
        st.download_button(
            "DOWNLOAD PER-DESCRIPTOR CSV",
            per_descriptor.to_csv(index=False),
            file_name="per_descriptor_auc.csv",
            mime="text/csv",
            key="dl_auc",
        )

    with tab_nose:
        classes = nose["classes"]
        matrix = pd.DataFrame(nose["confusion_matrix"], index=classes, columns=classes)
        st.subheader("Confusion matrix")
        st.caption("Rows = presented, columns = predicted.")
        _table(_heatmap(matrix), width="stretch")

        curve = pd.DataFrame(
            [
                {"noise": float(level), "accuracy": accuracy}
                for level, accuracy in nose["accuracy_by_noise"].items()
            ]
        ).sort_values("noise")
        st.subheader("Noise degradation")
        st.line_chart(curve.set_index("noise")["accuracy"], height=260)
        _table(curve, hide_index=True, width="stretch")
        st.json(
            {
                "seeds": nose["seeds"],
                "model": nose["model"],
                "train_seconds": nose["train_seconds"],
                "coffee_wood_smoke_confusion": nose.get("coffee_wood_smoke_confusion"),
            }
        )
        st.caption(
            "Nose accuracy is optimistic by design: train and test sessions come from "
            "the same simulator. See Documentation → Honesty."
        )


def _sample_library(ctx: pipeline.Context) -> None:
    _section("06 // SAMPLE LIBRARY")
    st.write("The six hand-authored aroma compositions behind the nose's classes.")
    overview = pd.DataFrame(
        [
            {
                "id": sample_id,
                "label": sample.label,
                "molecules": len(sample.molecules),
                "weight sum": float(sample.weights.sum()),
            }
            for sample_id, sample in ctx.samples.items()
        ]
    )
    _table(overview, hide_index=True, width="stretch")

    selected = st.selectbox(
        "Inspect sample",
        list(ctx.samples),
        format_func=lambda sample_id: ctx.samples[sample_id].label,
        key="library_sample",
    )
    sample = ctx.samples[selected]
    molecule_rows = []
    for molecule in sample.molecules:
        features = sensor.molecule_features(molecule["smiles"])
        molecule_rows.append(
            {
                "name": molecule["name"],
                "SMILES": molecule["smiles"],
                "weight": molecule["weight"],
                "MW": features[0],
                "logP": features[1],
                "O frac": features[2],
                "S": features[3],
                "N": features[4],
                "aromatic rings": features[5],
                "TPSA": features[6],
                "HBD": features[7],
            }
        )
    st.subheader(f"{sample.label} composition")
    _table(pd.DataFrame(molecule_rows), hide_index=True, width="stretch")
    st.bar_chart(
        pd.DataFrame(
            {"weight": sample.weights},
            index=[molecule["name"] for molecule in sample.molecules],
        ),
        height=240,
    )

    profile = pipeline.describe(ctx, selected, top_k=10)
    st.subheader("Descriptor profile")
    _table(pd.DataFrame(profile), hide_index=True, width="stretch")
    st.bar_chart(
        pd.DataFrame(
            {"score": [item["score"] for item in profile]},
            index=[item["descriptor"] for item in profile],
        ),
        height=260,
    )
    st.caption("Weights are illustrative relative amounts, not measured formulations.")
    st.download_button(
        "DOWNLOAD samples.json",
        SAMPLES_PATH.read_text(encoding="utf-8"),
        file_name="samples.json",
        mime="application/json",
        key="dl_samples",
    )

def _sniff_log() -> None:
    _section("07 // SNIFF LOG")
    st.write("Recent sniffs written to `logs/sniffs.jsonl`.")
    if not LOG_PATH.exists():
        st.info("No sniff log yet. Run a sniff in the Sniff Lab tab.")
        return

    records = []
    with LOG_PATH.open(encoding="utf-8") as handle:
        for line in deque(handle, maxlen=500):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        st.info("The log exists but contains no valid JSON lines.")
        return

    rows = []
    for record in records:
        conditions = record.get("conditions", {})
        nose = record.get("nose", {})
        rows.append(
            {
                "timestamp": record.get("timestamp", ""),
                "presented": record.get("presented", ""),
                "predicted": nose.get("predicted", ""),
                "confidence": nose.get("confidence", 0.0),
                "correct": nose.get("correct", False),
                "noise": conditions.get("noise"),
                "humidity": conditions.get("humidity"),
                "seed": conditions.get("seed"),
            }
        )
    df = pd.DataFrame(rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("LOG LINES", f"{len(df)}")
    c2.metric("ACCURACY", f"{df['correct'].mean():.1%}")
    c3.metric("LAST SNIFF", str(df["timestamp"].iloc[-1])[:19])

    choices = ["all"] + sorted(df["presented"].unique().tolist())
    selected = st.selectbox("Filter by presented sample", choices, key="log_filter")
    filtered = df if selected == "all" else df[df["presented"] == selected]
    _table(
        filtered.sort_values("timestamp", ascending=False).head(200),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "DOWNLOAD LOG CSV",
        filtered.to_csv(index=False),
        file_name="sniff_log.csv",
        mime="text/csv",
        key="dl_log_csv",
    )
    st.download_button(
        "DOWNLOAD LAST 500 JSONL",
        "\n".join(json.dumps(record) for record in records) + "\n",
        file_name="sniffs.jsonl",
        mime="application/jsonl",
        key="dl_log_jsonl",
    )
    st.caption("The log is runtime state, gitignored, and capped to the last 500 lines in this view.")


API_DOC = '''## Python API

### Load once

```python
import pipeline
ctx = pipeline.load_all()
```

`load_all()` validates the model files, loads the sample bank and both models, and precomputes the smell-model probabilities for every sample molecule.

### Run a sniff

```python
result = pipeline.sniff(ctx, "coffee", noise=0.03, humidity=0.5, seed=42)
result["nose"]["predicted"]       # "coffee"
result["nose"]["confidence"]      # top probability
result["sensors"]["values"]       # 8 readings
result["description"][0]          # {"descriptor", "score", "lift"}
```

`sniff()` appends to `logs/sniffs.jsonl` by default. Pass `log=False` for batch or sweep work.

### Describe a sample

```python
items = pipeline.describe(ctx, "mint", top_k=5)
```

Ranking is `score * sqrt(lift)`, with `lift = score / max(base_rate, 0.005)`.

### Describe one molecule

```python
items = pipeline.molecule_descriptors(ctx, "CC(C)C1CCC(C)CC1O", top_k=8)
```

Invalid SMILES raises `ValueError`.

### Low-level simulator

```python
import numpy as np
import sensor

readings = sensor.simulate(
    ctx.samples["lemon"].molecules,
    noise=0.03,
    humidity=0.5,
    rng=np.random.default_rng(42),
)
```

### Return shape

```json
{
  "presented": "coffee",
  "conditions": {"noise": 0.03, "humidity": 0.5, "seed": 42},
  "sensors": {"names": ["S1"], "values": [1.0]},
  "nose": {"predicted": "coffee", "confidence": 0.97, "probabilities": {}, "correct": true},
  "description": [{"descriptor": "roasted", "score": 0.61, "lift": 8.2}],
  "molecules": [{"name": "furfural", "smiles": "O=Cc1ccco1", "weight": 0.15}]
}
```
'''

HONESTY_DOC = '''## What is real

- The **OpenPOM dataset** is real: 4,983 molecules with 138 odor-descriptor labels, derived from Lee et al., *Science* (2023).
- The **smell model** is real: a Morgan-fingerprint RandomForest trained and evaluated on a held-out split. Its macro ROC-AUC is measured, not invented.
- The **sample recipes** are real aroma compounds, but the weights are illustrative, not measured formulations.

## What is simulated

- The **nose** is simulated. Its eight channels come from RDKit molecular properties, a seeded sensor bank, volatility, saturation, humidity, drift, and Gaussian noise. There is no gas sensor.
- The **nose accuracy** is optimistic by construction: training and test sessions come from the same simulator. High accuracy shows the pipeline works; it does not estimate accuracy on real hardware.
- The **noise curve** is the honest part: accuracy falls sharply as σ rises. The current curve is on the Model Metrics tab.

## What the descriptor scores mean

They are model probabilities for a molecule mix, averaged over the sample's molecules. They are not measurements of a real smell. Ranking uses lift over the dataset base rate so common terms like `sweet` do not dominate every sample.

## What the smell metric means

The 0.832 macro ROC-AUC uses a random 80/20 split. It may be optimistic for molecules with scaffolds unlike anything in training. A scaffold split is a documented improvement, not a claim made here.

## What this is not

- Not a safety, medical, or fragrance-formulation tool.
- Not a real electronic nose.
- Not a claim that the model can smell.
- Not a substitute for the OpenPOM authors' graph neural network, which reports a higher number in their tests.
'''


def _render_doc(path: Path) -> None:
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Missing documentation file: `{path.relative_to(ROOT)}`")


def _documentation() -> None:
    _section("08 // DOCUMENTATION")
    st.write(
        "Most documentation tabs render the repository markdown, so the app and the "
        "written specs cannot drift apart silently. API and Honesty are maintained inline."
    )
    docs = {
        "GUIDE": DOCS_DIR / "WEB_APP.md",
        "QUICKSTART": ROOT / "README.md",
        "ARCHITECTURE": DOCS_DIR / "TRD.md",
        "ALGORITHMS": DOCS_DIR / "ALGORITHMS.md",
        "SCHEMAS": DOCS_DIR / "BACKEND_SCHEMA.md",
        "APP FLOW": DOCS_DIR / "APP_FLOW.md",
        "PRODUCT": DOCS_DIR / "PRD.md",
        "DEPLOYMENT": DOCS_DIR / "DEPLOYMENT.md",
        "API": None,
        "HONESTY": None,
    }
    tabs = st.tabs(list(docs))
    for tab, (name, path) in zip(tabs, docs.items()):
        with tab:
            if name == "API":
                st.markdown(API_DOC)
            elif name == "HONESTY":
                st.markdown(HONESTY_DOC)
            else:
                _render_doc(path)


def main() -> None:
    if pipeline.missing_model_files():
        _bootstrap_screen()
        st.stop()

    metrics = _load_metrics()
    ctx = _load_context()
    picked, noise, humidity, seed, backend = _sidebar(ctx, metrics)
    _hero(loaded=True, backend=backend)

    tabs = st.tabs(
        [
            "SNIFF LAB",
            "BATCH ARENA",
            "MOLECULE LAB",
            "SENSOR SWEEP",
            "MODEL METRICS",
            "SAMPLE LIBRARY",
            "SNIFF LOG",
            "DOCUMENTATION",
        ]
    )
    with tabs[0]:
        _sniff_lab(ctx, picked, noise, humidity, seed, backend)
    with tabs[1]:
        _batch_arena(ctx, noise, humidity)
    with tabs[2]:
        _molecule_lab(ctx, backend)
    with tabs[3]:
        _sensor_sweep(ctx, noise, humidity)
    with tabs[4]:
        _model_metrics(metrics)
    with tabs[5]:
        _sample_library(ctx)
    with tabs[6]:
        _sniff_log()
    with tabs[7]:
        _documentation()

    st.markdown(
        '<div class="footer">SMELL//OS · NOSE: SIMULATED · DATA: OpenPOM · '
        "The eight channels are model-derived, not gas-sensor measurements. "
        "See Documentation → HONESTY.</div>",
        unsafe_allow_html=True,
    )


main()
