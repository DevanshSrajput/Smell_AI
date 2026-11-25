"""Headless end-to-end checks for the Smell AI pipeline (TRD.md §6)."""

from __future__ import annotations

import json
import sys

import numpy as np
from rdkit import Chem

import pipeline

N_SNIFFS = 20
MIN_SAMPLE_ACCURACY = 0.95
EXPECTED = {
    "mint": {"mint", "cooling", "herbal", "camphoreous", "minty"},
    "lemon": {"citrus", "lemon", "fresh", "terpenic"},
    "banana": {"fruity", "banana", "ester", "sweet"},
    "vanilla_candle": {"vanilla", "sweet", "creamy", "balsamic"},
    "coffee": {"roasted", "coffee", "nutty", "caramellic"},
    "wood_smoke": {"smoky", "phenolic", "woody", "burnt"},
}


def _fail(message: str) -> int:
    print(f"FAIL: {message}")
    return 1


def check_model_files() -> int:
    missing = pipeline.missing_model_files()
    if not missing:
        print("[1/7] model files: PASS")
        return 0
    print("[1/7] model files: FAIL")
    for rel in missing:
        print(f"  missing {rel}  (run: {pipeline.REQUIRED_FILES[rel]})")
    print("Train the models first:\n  python smell_model.py\n  python train_nose.py")
    return 1


def check_smiles(ctx: pipeline.Context) -> int:
    failures = []
    n_molecules = 0
    for sample in ctx.samples.values():
        for molecule in sample.molecules:
            n_molecules += 1
            if Chem.MolFromSmiles(molecule["smiles"]) is None:
                failures.append(f"{sample.id}/{molecule['name']}: {molecule['smiles']!r}")
    if failures:
        print("[2/7] sample SMILES: FAIL")
        for failure in failures:
            print(f"  unparsable {failure}")
        return 1
    print(
        f"[2/7] sample SMILES: PASS ({len(ctx.samples)} samples, {n_molecules} molecules)"
    )
    return 0


def check_sniff_keys(ctx: pipeline.Context) -> int:
    problems = []
    for sample_id in ctx.samples:
        result = pipeline.sniff(ctx, sample_id, seed=0)
        if result["presented"] != sample_id:
            problems.append(f"{sample_id}: presented is {result['presented']!r}")
        for key in ("presented", "conditions", "sensors", "nose", "description", "molecules"):
            if key not in result:
                problems.append(f"{sample_id}: missing key {key!r}")
        if len(result["sensors"]["values"]) != 8:
            problems.append(f"{sample_id}: sensors.values does not have length 8")
        if len(result["sensors"]["names"]) != 8:
            problems.append(f"{sample_id}: sensors.names does not have length 8")
        if len(result["description"]) != 5:
            problems.append(f"{sample_id}: description does not have 5 items")
        for item in result["description"]:
            if set(item) != {"descriptor", "score", "lift"}:
                problems.append(f"{sample_id}: bad description item: {item}")
            if item["descriptor"] not in ctx.descriptors:
                problems.append(f"{sample_id}: unknown descriptor {item['descriptor']!r}")
        if set(result["nose"]["probabilities"]) != set(ctx.samples):
            problems.append(f"{sample_id}: probabilities do not cover all samples")
        if result["nose"]["predicted"] not in ctx.samples:
            problems.append(f"{sample_id}: unknown predicted class")
        if not result["molecules"]:
            problems.append(f"{sample_id}: empty molecules list")
    if problems:
        print("[3/7] sniff() keys: FAIL")
        for problem in problems:
            print(f"  {problem}")
        print(json.dumps(pipeline.sniff(ctx, "coffee", seed=0), indent=2))
        return 1
    print(f"[3/7] sniff() keys: PASS (all {len(ctx.samples)} samples)")
    return 0


def check_nose_accuracy(ctx: pipeline.Context) -> int:
    classes = list(ctx.samples)
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    rows = []
    for i, sample_id in enumerate(classes):
        correct = 0
        for j in range(N_SNIFFS):
            result = pipeline.sniff(
                ctx, sample_id, noise=0.03, humidity=0.5, seed=1000 + i * 100 + j
            )
            predicted = result["nose"]["predicted"]
            confusion[i, classes.index(predicted)] += 1
            correct += predicted == sample_id
        accuracy = correct / N_SNIFFS
        rows.append((sample_id, correct, accuracy, accuracy >= MIN_SAMPLE_ACCURACY))

    print(f"[4/7] nose accuracy over {N_SNIFFS} sniffs per sample (noise 0.03)")
    print(f"  {'sample':15s} {'correct':>8s} {'accuracy':>9s}  result")
    for sample_id, correct, accuracy, passed in rows:
        print(
            f"  {sample_id:15s} {correct:>4d}/{N_SNIFFS:<3d} "
            f"{accuracy:>8.0%}  {'PASS' if passed else 'FAIL'}"
        )
    n_passing = sum(1 for _, _, _, passed in rows if passed)
    overall = sum(correct for _, correct, _, _ in rows) / (N_SNIFFS * len(classes))
    print(
        f"  overall {overall:.1%}; {n_passing}/{len(classes)} samples at or above 95%"
    )
    if n_passing < 5 or overall < MIN_SAMPLE_ACCURACY:
        print("  confusion matrix (rows=true, cols=pred):")
        print(f"  {'':15s} " + " ".join(f"{c[:6]:>6s}" for c in classes))
        for sample_id, row in zip(classes, confusion):
            print(f"  {sample_id:15s} " + " ".join(f"{value:>6d}" for value in row))
        return _fail(
            f"nose accuracy below target "
            f"({n_passing}/{len(classes)} samples, overall {overall:.1%})"
        )
    return 0


def check_descriptors(ctx: pipeline.Context) -> int:
    print("[5/7] descriptor sanity (top 5, score * sqrt(lift))")
    failures = []
    for sample_id, expected in EXPECTED.items():
        top = pipeline.describe(ctx, sample_id, top_k=5)
        names = [item["descriptor"] for item in top]
        valid_expected = expected & set(ctx.descriptors)
        if not valid_expected:
            failures.append(f"{sample_id}: expected set empty after filtering")
            print(f"  {sample_id:15s} {'FAIL':4s} expected names not in descriptors.json")
            continue
        hit = sorted(valid_expected & set(names))
        passed = bool(hit)
        print(f"  {sample_id:15s} {'PASS' if passed else 'FAIL'}  {', '.join(names)}")
        if not passed:
            failures.append(f"{sample_id}: none of {sorted(valid_expected)} in {names}")
    if failures:
        for failure in failures:
            print(f"  actual top 5 cause: {failure}")
        return _fail("descriptor expectations not met")
    return 0


def check_probabilities(ctx: pipeline.Context) -> int:
    result = pipeline.sniff(ctx, "mint", seed=7)
    total = sum(result["nose"]["probabilities"].values())
    if abs(total - 1.0) > 1e-6:
        return _fail(f"probabilities sum to {total!r}, not 1")
    print(f"[6/7] probabilities sum to 1: PASS (error {abs(total - 1.0):.2e})")
    return 0


def check_determinism(ctx: pipeline.Context) -> int:
    first = pipeline.sniff(ctx, "lemon", noise=0.05, humidity=0.4, seed=123)
    second = pipeline.sniff(ctx, "lemon", noise=0.05, humidity=0.4, seed=123)
    same_sensors = np.array_equal(first["sensors"]["values"], second["sensors"]["values"])
    same_nose = first["nose"] == second["nose"]
    same_description = first["description"] == second["description"]
    if not (same_sensors and same_nose and same_description):
        return _fail("same seed produced different readings or predictions")
    print("[7/7] determinism with fixed seed: PASS")
    return 0


def main() -> int:
    code = check_model_files()
    if code:
        return code
    ctx = pipeline.load_all()
    checks = (
        check_smiles,
        check_sniff_keys,
        check_nose_accuracy,
        check_descriptors,
        check_probabilities,
        check_determinism,
    )
    for check in checks:
        code = check(ctx)
        if code:
            return code
    print("PASS: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
