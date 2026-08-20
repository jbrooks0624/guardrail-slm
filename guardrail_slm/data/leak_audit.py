"""Demonstrate train/test separation with similarity distributions and overlap checks."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from guardrail_slm.data.config import SplitConfig, load_sources_config
from guardrail_slm.data.dedupe import clustering_text, minhash_of
from guardrail_slm.data.schema import AssignedRecord, read_assigned_records
from guardrail_slm.data.splits import SLICE_FILES, tactic_key
from guardrail_slm.settings import Settings, get_settings

EmbedFn = Callable[[Sequence[str]], np.ndarray]

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


class LeakError(ValueError):
    """Train/test overlap that must fail the build, not warn."""


def load_split_records(processed_dir: Path) -> list[AssignedRecord]:
    rows: list[AssignedRecord] = []
    for filename in SLICE_FILES.values():
        path = processed_dir / filename
        if path.is_file():
            rows.extend(read_assigned_records(path))
    if not rows:
        raise FileNotFoundError(f"no processed split JSONL under {processed_dir}")
    return rows


def load_held_out_tactic_sets(processed_dir: Path) -> tuple[tuple[str, ...], ...]:
    path = processed_dir / "split_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("held_out_tactic_sets") or []
    return tuple(tuple(str(item) for item in row) for row in raw)


def check_vanilla_overlap(records: Sequence[AssignedRecord]) -> None:
    train_ids = {
        record.vanilla_id
        for record in records
        if record.split in {"train", "val"} and record.vanilla_id
    }
    test_ids = {
        record.vanilla_id for record in records if record.split == "test" and record.vanilla_id
    }
    overlap = train_ids & test_ids
    if overlap:
        raise LeakError(f"{len(overlap)} vanilla_id values appear in both train/val and test")


def check_held_out_tactics(
    records: Sequence[AssignedRecord],
    held_out_tactic_sets: Sequence[tuple[str, ...]],
) -> None:
    held = set(held_out_tactic_sets)
    if not held:
        return
    leaked = sorted(
        {
            tactic_key(record)
            for record in records
            if record.split in {"train", "val"} and tactic_key(record) in held
        }
    )
    if leaked:
        raise LeakError(f"held-out tactic-set appeared in train/val: {leaked[:3]}")


def assert_no_leaks(
    records: Sequence[AssignedRecord],
    held_out_tactic_sets: Sequence[tuple[str, ...]],
) -> None:
    check_vanilla_overlap(records)
    check_held_out_tactics(records, held_out_tactic_sets)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def nearest_cosine(train: np.ndarray, test: np.ndarray, *, chunk: int = 256) -> np.ndarray:
    """Max cosine of each test row against train. Inputs should be L2-normalized."""
    nn = np.empty(len(test), dtype=np.float64)
    for start in range(0, len(test), chunk):
        block = test[start : start + chunk]
        nn[start : start + chunk] = (block @ train.T).max(axis=1)
    return nn


def minhash_matrix(texts: Sequence[str], *, num_perm: int, shingle_n: int) -> np.ndarray:
    rows = [minhash_of(text, num_perm=num_perm, shingle_n=shingle_n).hashvalues for text in texts]
    return np.stack(rows) if rows else np.zeros((0, num_perm), dtype=np.uint64)


def nearest_jaccard(
    train_hashes: np.ndarray, test_hashes: np.ndarray, *, chunk: int = 64
) -> np.ndarray:
    """Max estimated Jaccard of each test MinHash against train."""
    nn = np.empty(len(test_hashes), dtype=np.float64)
    n_train = len(train_hashes)
    train_chunk = 2048
    for start in range(0, len(test_hashes), chunk):
        block = test_hashes[start : start + chunk]
        best = np.zeros(len(block), dtype=np.float64)
        for train_start in range(0, n_train, train_chunk):
            train_block = train_hashes[train_start : train_start + train_chunk]
            equal = block[:, None, :] == train_block[None, :, :]
            scores = equal.mean(axis=2)
            best = np.maximum(best, scores.max(axis=1))
        nn[start : start + chunk] = best
    return nn


def default_embedder(model_name: str = DEFAULT_EMBEDDER) -> EmbedFn:
    """Load MiniLM once. Tests must inject a mock instead of calling this."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def _embed(texts: Sequence[str]) -> np.ndarray:
        encoded = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
            batch_size=64,
        )
        return np.asarray(encoded, dtype=np.float64)

    return _embed


def write_leak_plot(
    cosine_nn: np.ndarray,
    jaccard_nn: np.ndarray,
    *,
    cosine_threshold: float,
    jaccard_threshold: float,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cos = int((cosine_nn >= cosine_threshold).sum())
    n_jac = int((jaccard_nn >= jaccard_threshold).sum())
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(cosine_nn, bins=40, color="#3b6ea5", edgecolor="white")
    axes[0].axvline(
        cosine_threshold, color="#c0392b", linestyle="--", label=f"≥ {cosine_threshold}"
    )
    axes[0].set_title("Train-to-test NN cosine")
    axes[0].set_xlabel(f"{n_cos}/{len(cosine_nn)} test rows above threshold")
    axes[0].set_ylabel("test rows")
    axes[0].legend()
    axes[1].hist(jaccard_nn, bins=40, color="#4a8c5c", edgecolor="white")
    axes[1].axvline(
        jaccard_threshold, color="#c0392b", linestyle="--", label=f"≥ {jaccard_threshold}"
    )
    axes[1].set_title("Train-to-test NN Jaccard (MinHash)")
    axes[1].set_xlabel(f"{n_jac}/{len(jaccard_nn)} test rows above threshold")
    axes[1].legend()
    fig.suptitle("Leak audit: similarity is demonstrated, not asserted to be zero")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def leak_summary(
    records: Sequence[AssignedRecord],
    *,
    cosine_nn: np.ndarray,
    jaccard_nn: np.ndarray,
    cosine_threshold: float,
    jaccard_threshold: float,
    held_out_tactic_sets: Sequence[tuple[str, ...]],
) -> dict[str, object]:
    n_train = sum(record.split == "train" for record in records)
    n_test = sum(record.split == "test" for record in records)
    return {
        "n_train": n_train,
        "n_test": n_test,
        "cosine_threshold": cosine_threshold,
        "jaccard_threshold": jaccard_threshold,
        "n_test_cosine_above": int((cosine_nn >= cosine_threshold).sum()),
        "n_test_jaccard_above": int((jaccard_nn >= jaccard_threshold).sum()),
        "cosine_max": float(cosine_nn.max()) if len(cosine_nn) else 0.0,
        "jaccard_max": float(jaccard_nn.max()) if len(jaccard_nn) else 0.0,
        "held_out_tactic_sets": [list(item) for item in held_out_tactic_sets],
        "vanilla_overlap": 0,
    }


def run_leak_audit(
    records: Sequence[AssignedRecord],
    held_out_tactic_sets: Sequence[tuple[str, ...]],
    *,
    split_config: SplitConfig | None = None,
    embed_fn: EmbedFn | None = None,
    plot_dir: Path | None = None,
) -> dict[str, object]:
    split_config = split_config or load_sources_config().splits
    assert_no_leaks(records, held_out_tactic_sets)
    train = [record for record in records if record.split == "train"]
    test = [record for record in records if record.split == "test"]
    if not train or not test:
        raise ValueError("leak audit needs both train and test rows")

    embed = embed_fn or default_embedder(split_config.leak_embedder)
    texts = [record.text for record in train] + [record.text for record in test]
    if len(texts) > 1000:
        print("[leak-audit] embedding train and test prompts", file=sys.stderr, flush=True)
    embedded = l2_normalize(np.asarray(embed(texts), dtype=np.float64))
    train_emb = embedded[: len(train)]
    test_emb = embedded[len(train) :]
    cosine_nn = nearest_cosine(train_emb, test_emb)

    if len(train) + len(test) > 1000:
        print("[leak-audit] MinHash Jaccard nearest neighbors", file=sys.stderr, flush=True)
    train_h = minhash_matrix(
        [clustering_text(record) for record in train],
        num_perm=split_config.minhash_num_perm,
        shingle_n=split_config.shingle_n,
    )
    test_h = minhash_matrix(
        [clustering_text(record) for record in test],
        num_perm=split_config.minhash_num_perm,
        shingle_n=split_config.shingle_n,
    )
    jaccard_nn = nearest_jaccard(train_h, test_h)

    settings = get_settings()
    out_dir = plot_dir if plot_dir is not None else settings.results_dir / "plots"
    plot_path = out_dir / "leak_audit.png"
    json_path = out_dir / "leak_audit.json"
    write_leak_plot(
        cosine_nn,
        jaccard_nn,
        cosine_threshold=split_config.leak_cosine_threshold,
        jaccard_threshold=split_config.leak_jaccard_threshold,
        path=plot_path,
    )
    summary = leak_summary(
        records,
        cosine_nn=cosine_nn,
        jaccard_nn=jaccard_nn,
        cosine_threshold=split_config.leak_cosine_threshold,
        jaccard_threshold=split_config.leak_jaccard_threshold,
        held_out_tactic_sets=held_out_tactic_sets,
    )
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def audit_processed(
    settings: Settings | None = None, *, embed_fn: EmbedFn | None = None
) -> dict[str, object]:
    settings = settings or get_settings()
    catalog = load_sources_config(settings.sources_path)
    records = load_split_records(settings.processed_dir)
    held = load_held_out_tactic_sets(settings.processed_dir)
    return run_leak_audit(
        records,
        held,
        split_config=catalog.splits,
        embed_fn=embed_fn,
        plot_dir=settings.results_dir / "plots",
    )
