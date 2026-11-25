"""Train and serve an OpenPOM-style Message Passing Neural Network (MPNN) on GPU/CPU."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import smell_model

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
GNN_PATH = MODELS_DIR / "smell_gnn.pt"
GNN_METRICS_PATH = MODELS_DIR / "gnn_metrics.json"

ATOM_LIST = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _one_hot(val: Any, choices: list[Any]) -> list[float]:
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(val) if val in choices else len(choices)] = 1.0
    return vec


def _atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_LIST)
        + _one_hot(min(atom.GetTotalDegree(), 5), [0, 1, 2, 3, 4, 5])
        + _one_hot(max(-2, min(2, atom.GetFormalCharge())), [-2, -1, 0, 1, 2])
        + _one_hot(min(atom.GetTotalNumHs(), 4), [0, 1, 2, 3, 4])
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetMass() * 0.01,
        ]
    )


def _bond_features(bond: Chem.Bond) -> list[float]:
    return _one_hot(bond.GetBondType(), BOND_TYPES) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


NODE_DIM = len(_atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
EDGE_DIM = len(_bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def mol_to_graph(smiles: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = np.asarray([_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    edges: list[tuple[int, int]] = []
    eattr: list[list[float]] = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edges.extend([(u, v), (v, u)])
        eattr.extend([bf, bf])
    if not edges:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, EDGE_DIM), dtype=np.float32)
    else:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_attr = np.asarray(eattr, dtype=np.float32)
    return x, edge_index, edge_attr


def collate_graphs(
    graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    labels: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    xs, eis, eas, batch = [], [], [], []
    offset = 0
    for i, (x, ei, ea) in enumerate(graphs):
        n = x.shape[0]
        xs.append(torch.from_numpy(x))
        if ei.shape[1] > 0:
            eis.append(torch.from_numpy(ei) + offset)
            eas.append(torch.from_numpy(ea))
        batch.append(torch.full((n,), i, dtype=torch.long))
        offset += n
    out = {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.cat(eas, dim=0) if eas else torch.zeros((0, EDGE_DIM), dtype=torch.float32),
        "batch": torch.cat(batch, dim=0),
        "n_graphs": torch.tensor(len(graphs), dtype=torch.long),
    }
    if labels is not None:
        out["y"] = torch.from_numpy(labels.astype(np.float32))
    return out


class OpenPOMMPNN(nn.Module):
    """Edge-conditioned Message Passing Neural Network with attentive readout."""

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dim: int = 128,
        steps: int = 3,
        n_outputs: int = 138,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_outputs),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        h = self.node_proj(x)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            e = self.edge_net(edge_attr)
            for _ in range(self.steps):
                m = self.msg_mlp(torch.cat([h[src], e], dim=-1))
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, m)
                h = self.norm(self.gru(agg, h))

        # Global mean + attention-weighted pooling into a 256-dim POM embedding
        mean_pool = torch.zeros((n_graphs, h.shape[1]), device=h.device, dtype=h.dtype)
        counts = torch.zeros((n_graphs, 1), device=h.device, dtype=h.dtype)
        mean_pool.index_add_(0, batch, h)
        counts.index_add_(0, batch, torch.ones((h.shape[0], 1), device=h.device, dtype=h.dtype))
        mean_pool = mean_pool / counts.clamp(min=1.0)

        scores = self.attn_gate(h)
        exp_scores = torch.exp(scores - scores.max())
        denom = torch.zeros((n_graphs, 1), device=h.device, dtype=h.dtype)
        denom.index_add_(0, batch, exp_scores)
        weights = exp_scores / denom[batch].clamp(min=1e-8)
        attn_pool = torch.zeros_like(mean_pool)
        attn_pool.index_add_(0, batch, h * weights)

        pom_embed = torch.cat([mean_pool, attn_pool], dim=-1)
        return self.head(pom_embed)


def scaffold_split(smiles: list[str], test_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Bemis-Murcko scaffold split (no scaffold shared across train/test)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, sm in enumerate(smiles):
        mol = Chem.MolFromSmiles(sm)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else ""
        groups[scaf].append(idx)

    # Sort largest scaffold buckets first for balanced 80/20 sizing
    buckets = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n_test_target = int(round(len(smiles) * test_fraction))
    train_idx, test_idx = [], []
    for bucket in buckets:
        if len(test_idx) + len(bucket) <= n_test_target:
            test_idx.extend(bucket)
        else:
            train_idx.extend(bucket)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def _eval_auc(Y_true: np.ndarray, P_pred: np.ndarray, label_cols: list[str]) -> tuple[float, dict[str, float]]:
    per_auc: dict[str, float] = {}
    for j, name in enumerate(label_cols):
        if len(np.unique(Y_true[:, j])) == 2:
            per_auc[name] = float(roc_auc_score(Y_true[:, j], P_pred[:, j]))
    macro = float(np.mean(list(per_auc.values()))) if per_auc else 0.0
    return macro, per_auc


def _predict_graphs(
    model: OpenPOMMPNN,
    graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            batch = collate_graphs(graphs[i : i + batch_size])
            logits = model(
                batch["x"].to(device),
                batch["edge_index"].to(device),
                batch["edge_attr"].to(device),
                batch["batch"].to(device),
                int(batch["n_graphs"]),
            )
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(preds)


def _fit_gnn(
    graphs_train: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    Y_train: np.ndarray,
    graphs_val: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    Y_val: np.ndarray,
    label_cols: list[str],
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> tuple[OpenPOMMPNN, float, dict[str, float], list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = OpenPOMMPNN(n_outputs=len(label_cols)).to(device)
    pos = Y_train.sum(axis=0).astype(np.float32)
    neg = (len(Y_train) - pos).astype(np.float32)
    pos_weight = torch.from_numpy(np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = -1.0
    best_per_auc: dict[str, float] = {}
    best_state = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(graphs_train))
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = collate_graphs([graphs_train[i] for i in idx], Y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["x"].to(device),
                batch["edge_index"].to(device),
                batch["edge_attr"].to(device),
                batch["batch"].to(device),
                int(batch["n_graphs"]),
            )
            loss = criterion(logits, batch["y"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            P_val = _predict_graphs(model, graphs_val, device)
            val_auc, per_auc = _eval_auc(Y_val, P_val, label_cols)
            avg_loss = total_loss / max(n_batches, 1)
            history.append({"epoch": epoch, "loss": round(avg_loss, 4), "val_auc": round(val_auc, 4)})
            print(f"  epoch {epoch:02d}/{epochs} | loss={avg_loss:.4f} | val_macro_auc={val_auc:.4f}")
            if val_auc > best_auc:
                best_auc = val_auc
                best_per_auc = per_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, best_per_auc, history


def train(csv_path: str | None = None, epochs: int = 40, seed: int = 42) -> dict:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Training OpenPOM MPNN on {device_name} ({device})")

    smiles_raw, Y_raw, label_cols = smell_model.load_dataset(csv_path)
    graphs, kept_smiles, kept_idx = [], [], []
    for i, sm in enumerate(smiles_raw):
        g = mol_to_graph(sm)
        if g is not None:
            graphs.append(g)
            kept_smiles.append(sm)
            kept_idx.append(i)
    Y = Y_raw[kept_idx]

    # 1. Random 80/20 split (same seed=42 as smell_model.py for direct comparison)
    idx_all = np.arange(len(graphs))
    tr_rand, te_rand = train_test_split(idx_all, test_size=0.2, random_state=seed)
    print("Evaluating Random 80/20 split...")
    gnn_rand, rand_auc, rand_per_auc, rand_hist = _fit_gnn(
        [graphs[i] for i in tr_rand],
        Y[tr_rand],
        [graphs[i] for i in te_rand],
        Y[te_rand],
        label_cols,
        device,
        epochs=epochs,
        seed=seed,
    )

    # 2. Bemis-Murcko Scaffold 80/20 split (paper-rigor generalization check)
    tr_scaf, te_scaf = scaffold_split(kept_smiles, test_fraction=0.2)
    print(f"Evaluating Bemis-Murcko Scaffold split (train={len(tr_scaf)}, test={len(te_scaf)})...")
    _, scaf_auc, _, _ = _fit_gnn(
        [graphs[i] for i in tr_scaf],
        Y[tr_scaf],
        [graphs[i] for i in te_scaf],
        Y[te_scaf],
        label_cols,
        device,
        epochs=epochs,
        seed=seed,
    )

    # Also score RandomForest on the exact same scaffold split for an honest comparison
    X_fp, _ = smell_model.featurize(kept_smiles)
    rf_scaf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=seed, class_weight="balanced_subsample"
    )
    rf_scaf.fit(X_fp[tr_scaf], Y[tr_scaf])
    rf_scaf_auc, _ = _eval_auc(Y[te_scaf], smell_model._proba_matrix(rf_scaf, X_fp[te_scaf]), label_cols)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in gnn_rand.parameters())
    torch.save(
        {
            "state_dict": gnn_rand.state_dict(),
            "descriptors": label_cols,
            "node_dim": NODE_DIM,
            "edge_dim": EDGE_DIM,
            "hidden_dim": 128,
            "steps": 3,
        },
        GNN_PATH,
    )

    elapsed = round(time.perf_counter() - t0, 2)
    metrics = {
        "model": {
            "type": "OpenPOMMPNN",
            "hidden_dim": 128,
            "message_steps": 3,
            "n_parameters": n_params,
            "epochs": epochs,
        },
        "device": device_name,
        "cuda_used": device.type == "cuda",
        "n_molecules": len(graphs),
        "n_descriptors": len(label_cols),
        "random_split_macro_roc_auc": rand_auc,
        "scaffold_split_macro_roc_auc": scaf_auc,
        "rf_scaffold_split_macro_roc_auc": rf_scaf_auc,
        "n_descriptors_scored": len(rand_per_auc),
        "per_descriptor_auc": rand_per_auc,
        "history": rand_hist,
        "train_seconds": elapsed,
        "seed": seed,
    }
    GNN_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"GNN Random AUC:   {rand_auc:.4f}")
    print(f"GNN Scaffold AUC: {scaf_auc:.4f} (RF Scaffold AUC: {rf_scaf_auc:.4f})")
    print(f"Saved {GNN_PATH} and {GNN_METRICS_PATH} in {elapsed:.1f}s")
    return metrics


def load_gnn(path: Path = GNN_PATH) -> tuple[OpenPOMMPNN, list[str]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = OpenPOMMPNN(
        node_dim=ckpt["node_dim"],
        edge_dim=ckpt["edge_dim"],
        hidden_dim=ckpt["hidden_dim"],
        steps=ckpt["steps"],
        n_outputs=len(ckpt["descriptors"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["descriptors"]


def predict_proba(model: OpenPOMMPNN, smiles: list[str]) -> np.ndarray:
    graphs = []
    for sm in smiles:
        g = mol_to_graph(sm)
        if g is None:
            raise ValueError(f"Could not parse SMILES: {sm!r}")
        graphs.append(g)
    return _predict_graphs(model, graphs, torch.device("cpu"))


if __name__ == "__main__":
    train()
