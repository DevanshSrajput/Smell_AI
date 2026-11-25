"""Streamlit UI: pick a smell, sniff it with the simulated nose, describe it."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import pipeline

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


@st.cache_resource
def _load() -> pipeline.Context:
    return pipeline.load_all()


@st.cache_data
def _metrics() -> dict:
    metrics = {}
    for name in ("smell_metrics", "nose_metrics"):
        path = MODELS_DIR / f"{name}.json"
        if path.exists():
            metrics[name] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def _show_description(items: list[dict]) -> None:
    for item in items:
        st.write(
            f"**{item['descriptor']}** · score {item['score']:.2f} · "
            f"lift {item['lift']:.1f}×"
        )
        st.progress(float(min(max(item["score"], 0.0), 1.0)))


st.set_page_config(page_title="Smell AI", page_icon="👃", layout="wide")
st.title("Smell AI")
st.caption("A simulated electronic nose plus a molecule smell model. No hardware, no API keys.")

try:
    ctx = _load()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

sample_ids = list(ctx.samples)
metrics = _metrics()

with st.sidebar:
    st.header("Sniff conditions")
    noise = st.slider("Noise", 0.00, 0.30, 0.03, 0.01)
    humidity = st.slider("Humidity", 0.0, 1.0, 0.5, 0.05)
    seed = st.number_input(
        "Seed (blank = fresh reading)",
        min_value=0,
        value=None,
        step=1,
        placeholder="blank",
    )
    st.divider()
    with st.expander("Measured model metrics"):
        if "smell_metrics" in metrics:
            st.write(
                f"Smell model macro ROC-AUC: "
                f"{metrics['smell_metrics']['macro_roc_auc']:.3f}"
            )
        if "nose_metrics" in metrics:
            st.write(
                f"Nose held-out accuracy: "
                f"{metrics['nose_metrics']['accuracy_default_noise']:.1%}"
            )
        st.caption("Numbers come from the saved metrics JSON; the nose is simulated.")

tab_sniff, tab_molecule = st.tabs(["Pick then Sniff", "Single molecule"])

with tab_sniff:
    picked = st.selectbox(
        "Smell", sample_ids, format_func=lambda sample_id: ctx.samples[sample_id].label
    )
    if st.button("Sniff", type="primary"):
        try:
            st.session_state["last"] = pipeline.sniff(
                ctx, picked, noise=noise, humidity=humidity, seed=seed
            )
        except (KeyError, ValueError) as exc:
            st.error(str(exc))

    result = st.session_state.get("last")
    if result is None:
        st.write("Pick a smell and press Sniff.")
    else:
        presented = ctx.samples[result["presented"]].label
        predicted = result["nose"]["predicted"]
        predicted_label = ctx.samples[predicted].label
        confidence = result["nose"]["confidence"]

        st.subheader("Sensor channels")
        st.bar_chart(
            pd.DataFrame(
                {"reading": result["sensors"]["values"]},
                index=result["sensors"]["names"],
            ),
            height=260,
        )
        st.caption("Readings are simulated. 1.0 means no change from clean air.")

        left, right = st.columns([1, 2])
        with left:
            st.subheader("Nose guess")
            if confidence < 0.5:
                st.markdown("### Low confidence")
                st.write(f"Top guess: {predicted_label} ({confidence:.0%})")
            else:
                st.markdown(f"### {predicted_label}")
                st.write(f"Confidence {confidence:.0%}")
            if not result["nose"]["correct"]:
                st.warning(f"Nose guessed {predicted_label}. You presented {presented}.")
        with right:
            st.subheader("Probabilities")
            st.bar_chart(
                pd.DataFrame(
                    {
                        "probability": [
                            result["nose"]["probabilities"][sample_id]
                            for sample_id in sample_ids
                        ]
                    },
                    index=[ctx.samples[sample_id].label for sample_id in sample_ids],
                ),
                height=260,
            )

        st.subheader(f"Smell description: {predicted_label}")
        _show_description(result["description"])
        st.caption(
            "Scores come from the molecule model, averaged over this sample's aroma "
            "molecules. They are model scores, not measurements."
        )

        with st.expander("Molecules behind this description"):
            st.dataframe(
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
            )

with tab_molecule:
    st.subheader("Describe one molecule")
    smiles = st.text_input(
        "SMILES",
        value="CC(C)C1CCC(C)CC1O",
        help="Example default: menthol.",
    )
    if st.button("Describe molecule"):
        try:
            st.session_state["molecule"] = pipeline.molecule_descriptors(ctx, smiles)
            st.session_state["molecule_smiles"] = smiles
        except ValueError as exc:
            st.session_state.pop("molecule", None)
            st.error(str(exc))

    molecule_items = st.session_state.get("molecule")
    if molecule_items:
        st.caption(f"Top descriptors for `{st.session_state['molecule_smiles']}`")
        _show_description(molecule_items)
        st.caption("Structure-only model scores; not a measurement of a real smell.")

st.divider()
st.caption(
    "The nose is simulated: eight RDKit-derived channels, not gas sensors. "
    "See README Part 3 for what a real BME688 nose would require."
)
