from __future__ import annotations

import hashlib
import json
import os
import random
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


SPLIT_TYPES = ("random", "cold_drug", "cold_target", "all_cold", "scaffold")
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)
SPLIT_SUMMARY_FIELDNAMES = [
    "split",
    "num_samples",
    "num_unique_drugs",
    "num_unique_targets",
    "num_unique_pairs",
    "num_unique_scaffolds",
]


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(path, payload):
    create_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_csv(path, rows: List[Dict[str, object]], fieldnames: Sequence[str] | None = None):
    import csv

    create_dir(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_default_data_root():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get_default_results_root():
    return os.path.join(get_repo_root(), "results")


def fill_default_paths(args):
    if getattr(args, "data_root", None) is None:
        args.data_root = get_default_data_root()
    if getattr(args, "results_root", None) is None:
        args.results_root = get_default_results_root()


def resolve_run_dir(results_root, dataset, model_name, model_variant, split_type, split_seed, train_seed, run_name=None, overwrite=False):
    if split_seed == train_seed:
        seed_name = f"seed_{train_seed}"
    else:
        seed_name = f"seed_{train_seed}__split_{split_seed}"

    seed_dir = os.path.join(
        results_root,
        dataset,
        model_name,
        model_variant,
        split_type,
        seed_name,
    )

    run_dir = seed_dir if run_name is None else os.path.join(seed_dir, run_name)
    if os.path.exists(run_dir) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing run directory: {run_dir}. "
            "Use a new run name or pass --overwrite."
        )

    create_dir(run_dir)
    create_dir(os.path.join(run_dir, "checkpoints"))
    create_dir(os.path.join(run_dir, "artifacts"))

    return run_dir


def load_raw_tables(dataset_root):
    raw_dir = os.path.join(dataset_root, "raw")
    train_df = pd.read_csv(os.path.join(raw_dir, "data_train.csv")).copy()
    test_df = pd.read_csv(os.path.join(raw_dir, "data_test.csv")).copy()

    train_df["source_split"] = "legacy_train"
    test_df["source_split"] = "legacy_test"

    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    merged_df["global_index"] = np.arange(len(merged_df))
    merged_df["drug_id"] = merged_df["compound_iso_smiles"].astype(str)
    merged_df["target_id"] = merged_df["target_sequence"].astype(str)
    merged_df["pair_id"] = merged_df["drug_id"] + "||" + merged_df["target_id"]
    scaffold_rows = [murcko_scaffold(smiles) for smiles in merged_df["compound_iso_smiles"].astype(str)]
    merged_df["canonical_smiles"] = [row[1]["canonical_smiles"] or row[0] for row in scaffold_rows]
    merged_df["scaffold_id"] = [row[0] for row in scaffold_rows]
    merged_df["scaffold_fallback"] = [row[1]["scaffold_fallback"] for row in scaffold_rows]
    merged_df["canonicalization_failed"] = [bool(row[1]["canonicalization_failed"]) for row in scaffold_rows]

    return train_df, test_df, merged_df


def dataframe_sha256(df: pd.DataFrame) -> str:
    payload = df.sort_values("global_index").to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seeds(seeds_value: str | Sequence[int]) -> List[int]:
    if isinstance(seeds_value, str):
        return [int(item.strip()) for item in seeds_value.split(",") if item.strip()]
    return [int(item) for item in seeds_value]


def parse_fraction_string(value: str) -> Tuple[float, float, float]:
    fractions = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(fractions) != 3:
        raise ValueError("Fractions must contain exactly three comma-separated values.")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Fractions must sum to 1.0, got {fractions}")
    return fractions


def stable_hash(text: str, prefix: str, length: int = 16) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}::{digest}"


def canonicalize_smiles(smiles: str) -> tuple[str, dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        fallback = stable_hash(smiles, "INVALID_SMILES")
        return fallback, {
            "canonical_smiles": None,
            "canonicalization_failed": True,
            "scaffold_fallback": "invalid_smiles",
            "scaffold_id": fallback,
        }

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return canonical_smiles, {
        "canonical_smiles": canonical_smiles,
        "canonicalization_failed": False,
        "scaffold_fallback": None,
        "scaffold_id": None,
    }


def murcko_scaffold(smiles: str) -> tuple[str, dict]:
    canonical_smiles, metadata = canonicalize_smiles(smiles)
    if metadata["canonicalization_failed"]:
        return canonical_smiles, metadata

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=canonical_smiles)
    if scaffold:
        metadata["scaffold_id"] = scaffold
        return scaffold, metadata

    fallback = stable_hash(canonical_smiles, "NO_SCAFFOLD")
    metadata["scaffold_fallback"] = "empty_murcko"
    metadata["scaffold_id"] = fallback
    return fallback, metadata


def _split_file_candidates(dataset_root, split_type, seed):
    split_dir = os.path.join(dataset_root, "splits", split_type)
    return [
        os.path.join(split_dir, f"seed_{seed}.json"),
        os.path.join(split_dir, f"seed_{seed:03d}.json"),
        os.path.join(split_dir, f"seed_{seed:04d}.json"),
    ]


def split_manifest_path(dataset_root, split_type, seed):
    return _split_file_candidates(dataset_root, split_type, seed)[0]


def _load_explicit_split_file(dataset_root, split_type, seed):
    for candidate in _split_file_candidates(dataset_root, split_type, seed):
        if os.path.exists(candidate):
            payload = load_json(candidate)
            payload["_split_file"] = candidate
            return payload
    return None


def _normalize_split_payload(split_payload):
    normalized = {}
    for key in ("train", "val", "test"):
        if key not in split_payload:
            raise KeyError(f"Split payload is missing required key: {key}")
        normalized[key] = [int(idx) for idx in split_payload[key]]
    metadata = split_payload.get("metadata", {})
    audit = split_payload.get("audit", {})
    validation = split_payload.get("validation", {})
    return normalized, metadata, audit, validation


def assign_rowwise(indices: Sequence[int], fractions: Sequence[float], seed: int):
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices, dtype=int).copy()
    rng.shuffle(shuffled)
    train_end = int(round(len(shuffled) * fractions[0]))
    val_end = train_end + int(round(len(shuffled) * fractions[1]))
    train_idx = sorted(shuffled[:train_end].tolist())
    val_idx = sorted(shuffled[train_end:val_end].tolist())
    test_idx = sorted(shuffled[val_end:].tolist())
    return train_idx, val_idx, test_idx


def build_group_index(df: pd.DataFrame, key_column: str) -> Dict[str, List[int]]:
    grouped = df.groupby(key_column)["global_index"].apply(list)
    return {str(key): list(map(int, values)) for key, values in grouped.items()}


def assign_rows_from_groups(group_to_rows: Dict[str, List[int]], fractions: Sequence[float], seed: int):
    total_rows = sum(len(rows) for rows in group_to_rows.values())
    targets = [fraction * total_rows for fraction in fractions]
    rng = np.random.default_rng(seed)
    items = list(group_to_rows.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)

    buckets = [[], [], []]
    counts = [0, 0, 0]
    for _, rows in items:
        deficits = [targets[idx] - counts[idx] for idx in range(3)]
        chosen = int(np.argmax(deficits))
        buckets[chosen].extend(rows)
        counts[chosen] += len(rows)

    return tuple(sorted(bucket) for bucket in buckets)


def assign_group_keys(group_to_rows: Dict[str, List[int]], fractions: Sequence[float], seed: int):
    total_rows = sum(len(rows) for rows in group_to_rows.values())
    targets = [fraction * total_rows for fraction in fractions]
    rng = np.random.default_rng(seed)
    items = list(group_to_rows.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)

    buckets = [[], [], []]
    counts = [0, 0, 0]
    for key, rows in items:
        deficits = [targets[idx] - counts[idx] for idx in range(3)]
        chosen = int(np.argmax(deficits))
        buckets[chosen].append(key)
        counts[chosen] += len(rows)

    return tuple(sorted(bucket) for bucket in buckets)


def pairwise_overlap(sets: Dict[str, Iterable[str | int]]) -> Dict[str, int]:
    resolved = {name: set(values) for name, values in sets.items()}
    overlaps = {}
    for left, right in combinations(resolved.keys(), 2):
        overlaps[f"{left}__{right}"] = len(resolved[left] & resolved[right])
    return overlaps


def audit_split_indices(merged_df: pd.DataFrame, split_indices: Dict[str, List[int]]) -> Dict[str, object]:
    split_frames = {
        split_name: merged_df.iloc[indices].copy()
        for split_name, indices in split_indices.items()
    }
    return {
        "sizes": {split_name: int(len(frame)) for split_name, frame in split_frames.items()},
        "row_overlap": pairwise_overlap({name: frame["global_index"].tolist() for name, frame in split_frames.items()}),
        "pair_overlap": pairwise_overlap({name: frame["pair_id"].tolist() for name, frame in split_frames.items()}),
        "drug_overlap": pairwise_overlap({name: frame["drug_id"].tolist() for name, frame in split_frames.items()}),
        "target_overlap": pairwise_overlap({name: frame["target_id"].tolist() for name, frame in split_frames.items()}),
        "scaffold_overlap": pairwise_overlap({name: frame["scaffold_id"].tolist() for name, frame in split_frames.items()}),
    }


def expected_zero_overlap(split_type: str) -> Dict[str, List[str]]:
    expected = {
        "row_overlap": ["train__val", "train__test", "val__test"],
        "pair_overlap": ["train__val", "train__test", "val__test"],
    }
    if split_type == "cold_drug":
        expected["drug_overlap"] = ["train__val", "train__test", "val__test"]
    elif split_type == "cold_target":
        expected["target_overlap"] = ["train__val", "train__test", "val__test"]
    elif split_type == "all_cold":
        expected["drug_overlap"] = ["train__val", "train__test", "val__test"]
        expected["target_overlap"] = ["train__val", "train__test", "val__test"]
    elif split_type == "scaffold":
        expected["drug_overlap"] = ["train__val", "train__test", "val__test"]
        expected["scaffold_overlap"] = ["train__val", "train__test", "val__test"]
    return expected


def validate_split_audit(split_type: str, audit: Dict[str, object]) -> Dict[str, object]:
    empty_splits = [name for name, size in audit["sizes"].items() if int(size) == 0]
    violations = []
    for section, keys in expected_zero_overlap(split_type).items():
        values = audit.get(section, {})
        for key in keys:
            count = int(values.get(key, 0))
            if count != 0:
                violations.append({"section": section, "pair": key, "count": count})
    return {"ok": not violations and not empty_splits, "violations": violations, "empty_splits": empty_splits}


def generate_random_split(merged_df: pd.DataFrame, fractions: Sequence[float], seed: int):
    groups = build_group_index(merged_df, "pair_id")
    train_idx, val_idx, test_idx = assign_rows_from_groups(groups, fractions, seed)
    return {"train": train_idx, "val": val_idx, "test": test_idx}, {"dropped_rows": 0, "grouped_by": "pair_id"}


def generate_group_split(merged_df: pd.DataFrame, key_column: str, fractions: Sequence[float], seed: int):
    groups = build_group_index(merged_df, key_column)
    train_idx, val_idx, test_idx = assign_rows_from_groups(groups, fractions, seed)
    return {"train": train_idx, "val": val_idx, "test": test_idx}, {"dropped_rows": 0}


def generate_all_cold_split(merged_df: pd.DataFrame, fractions: Sequence[float], seed: int):
    drug_groups = build_group_index(merged_df, "drug_id")
    target_groups = build_group_index(merged_df, "target_id")

    train_drugs, val_drugs, test_drugs = assign_group_keys(drug_groups, fractions, seed)
    train_targets, val_targets, test_targets = assign_group_keys(target_groups, fractions, seed + 997)

    bucket_sets = {
        "train": (set(train_drugs), set(train_targets)),
        "val": (set(val_drugs), set(val_targets)),
        "test": (set(test_drugs), set(test_targets)),
    }
    split_indices = {"train": [], "val": [], "test": []}
    dropped_rows = 0
    for row in merged_df.itertuples(index=False):
        assigned = None
        for split_name, (drug_bucket, target_bucket) in bucket_sets.items():
            if row.drug_id in drug_bucket and row.target_id in target_bucket:
                assigned = split_name
                break
        if assigned is None:
            dropped_rows += 1
            continue
        split_indices[assigned].append(int(row.global_index))

    for split_name in split_indices:
        split_indices[split_name] = sorted(split_indices[split_name])
    return split_indices, {"dropped_rows": dropped_rows}


def generate_split_indices(merged_df: pd.DataFrame, split_type: str, seed: int, fractions: Sequence[float]):
    split_type = split_type.lower()
    if split_type == "random":
        return generate_random_split(merged_df, fractions, seed)
    if split_type == "cold_drug":
        return generate_group_split(merged_df, "drug_id", fractions, seed)
    if split_type == "cold_target":
        return generate_group_split(merged_df, "target_id", fractions, seed)
    if split_type == "all_cold":
        return generate_all_cold_split(merged_df, fractions, seed)
    if split_type == "scaffold":
        return generate_group_split(merged_df, "scaffold_id", fractions, seed)
    raise ValueError(f"Unsupported split type: {split_type}")


def scaffold_audit_summary(merged_df: pd.DataFrame) -> Dict[str, object]:
    fallback_rows = merged_df[merged_df["scaffold_fallback"].notna()].copy()
    fallback_counts = (
        fallback_rows["scaffold_fallback"].value_counts(dropna=False).to_dict()
        if len(fallback_rows) > 0
        else {}
    )
    return {
        "num_rows": int(len(merged_df)),
        "num_unique_scaffolds": int(merged_df["scaffold_id"].nunique()),
        "num_canonicalization_failures": int(merged_df["canonicalization_failed"].sum()),
        "num_scaffold_fallback_rows": int(len(fallback_rows)),
        "fallback_counts": {str(k): int(v) for k, v in fallback_counts.items()},
    }


def build_split_manifest(dataset_root: str, split_type: str, seed: int, fractions: Sequence[float] = DEFAULT_FRACTIONS):
    _, _, merged_df = load_raw_tables(dataset_root)
    split_indices, extras = generate_split_indices(merged_df, split_type, seed, fractions)
    validate_split_indices(split_indices, len(merged_df))
    audit = audit_split_indices(merged_df, split_indices)
    validation = validate_split_audit(split_type, audit)
    if not validation["ok"]:
        raise AssertionError(f"Generated split failed overlap validation: {validation}")
    raw_dir = Path(dataset_root) / "raw"
    split_summary = summarize_splits(merged_df, split_indices)
    manifest = {
        "train": split_indices["train"],
        "val": split_indices["val"],
        "test": split_indices["test"],
        "metadata": {
            "dataset_root": str(Path(dataset_root).resolve()),
            "split_type": split_type,
            "seed": int(seed),
            "fractions": list(map(float, fractions)),
            "source_files": {
                "data_train.csv": str((raw_dir / "data_train.csv").resolve()),
                "data_test.csv": str((raw_dir / "data_test.csv").resolve()),
            },
            "source_hashes": {
                "data_train.csv": file_sha256(raw_dir / "data_train.csv"),
                "data_test.csv": file_sha256(raw_dir / "data_test.csv"),
                "merged_dataframe": dataframe_sha256(merged_df),
            },
            "extras": extras,
            "scaffold_audit": scaffold_audit_summary(merged_df),
        },
        "audit": audit,
        "validation": validation,
        "summary": split_summary,
    }
    return manifest


def split_summary_rows(manifest: Dict[str, object]) -> List[Dict[str, object]]:
    summary = manifest["summary"]["splits"]
    rows = []
    for split_name in ("train", "val", "test"):
        payload = summary[split_name]
        rows.append(
            {
                "split": split_name,
                "num_samples": payload["num_samples"],
                "num_unique_drugs": payload["num_unique_drugs"],
                "num_unique_targets": payload["num_unique_targets"],
                "num_unique_pairs": payload["num_unique_pairs"],
                "num_unique_scaffolds": payload["num_unique_scaffolds"],
            }
        )
    return rows


def audit_rows(manifest: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for section, payload in manifest["audit"].items():
        if isinstance(payload, dict):
            for key, value in payload.items():
                rows.append({"section": section, "name": key, "value": value})
        else:
            rows.append({"section": "audit", "name": section, "value": payload})
    for key, value in manifest["validation"].items():
        if isinstance(value, list):
            rows.append({"section": "validation", "name": key, "value": json.dumps(value, sort_keys=True)})
        else:
            rows.append({"section": "validation", "name": key, "value": value})
    return rows


def save_split_manifest(dataset_root: str, split_type: str, seed: int, manifest: Dict[str, object], overwrite=False):
    path = split_manifest_path(dataset_root, split_type, seed)
    create_dir(os.path.dirname(path))
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite split manifest: {path}")
    save_json(path, manifest)
    summary_csv = path.replace(".json", ".summary.csv")
    audit_json = path.replace(".json", ".audit.json")
    audit_csv = path.replace(".json", ".audit.csv")
    save_csv(summary_csv, split_summary_rows(manifest), fieldnames=SPLIT_SUMMARY_FIELDNAMES)
    save_json(audit_json, {"audit": manifest["audit"], "validation": manifest["validation"], "summary": manifest["summary"]})
    save_csv(audit_csv, audit_rows(manifest), fieldnames=["section", "name", "value"])
    return path


def build_random_legacy_split(dataset_root, seed, val_fraction=0.1):
    train_df, test_df, merged_df = load_raw_tables(dataset_root)
    train_count = len(train_df)
    total_count = len(merged_df)

    if train_count < 2:
        raise ValueError("Need at least 2 legacy training samples to build a validation split.")

    legacy_train_df = merged_df.iloc[:train_count].copy()
    groups = build_group_index(legacy_train_df, "pair_id")
    fit_indices, val_indices, _ = assign_rows_from_groups(
        groups,
        (1.0 - val_fraction, val_fraction, 0.0),
        seed,
    )
    test_indices = [int(idx) for idx in range(train_count, total_count)]

    audit = audit_split_indices(merged_df, {"train": fit_indices, "val": val_indices, "test": test_indices})
    metadata = {
        "source": "legacy_random_holdout",
        "seed": seed,
        "val_fraction": val_fraction,
        "grouped_by": "pair_id",
    }
    validation = validate_split_audit("random", audit)
    if not validation["ok"]:
        raise AssertionError(f"Legacy random fallback split failed overlap validation: {validation}")
    return {"train": fit_indices, "val": val_indices, "test": test_indices}, metadata, audit, validation


def validate_split_indices(split_indices, total_count):
    seen = set()
    for split_name in ("train", "val", "test"):
        indices = split_indices[split_name]
        if not indices:
            raise ValueError(f"Split '{split_name}' is empty.")
        for idx in indices:
            if idx < 0 or idx >= total_count:
                raise IndexError(f"Index {idx} in split '{split_name}' is out of range for dataset size {total_count}.")
            if idx in seen:
                raise ValueError(f"Index {idx} appears in more than one split.")
            seen.add(idx)


def load_split_indices(dataset_root, split_type, seed, val_fraction=0.1):
    if split_type not in SPLIT_TYPES:
        raise ValueError(f"Unsupported split type: {split_type}. Expected one of {SPLIT_TYPES}.")

    _, _, merged_df = load_raw_tables(dataset_root)
    explicit_payload = _load_explicit_split_file(dataset_root, split_type, seed)

    if explicit_payload is not None:
        split_indices, metadata, audit, validation = _normalize_split_payload(explicit_payload)
    elif split_type == "random":
        split_indices, metadata, audit, validation = build_random_legacy_split(dataset_root, seed, val_fraction=val_fraction)
    else:
        expected_dir = os.path.join(dataset_root, "splits", split_type)
        raise FileNotFoundError(
            f"Split type '{split_type}' requires a pre-generated split file under {expected_dir}."
        )

    validate_split_indices(split_indices, len(merged_df))
    metadata = dict(metadata)
    if explicit_payload is not None:
        metadata["source"] = metadata.get("source", "manifest")
        metadata["split_file"] = explicit_payload["_split_file"]
    return split_indices, metadata, merged_df, audit, validation


def summarize_splits(merged_df, split_indices):
    split_frames = {}
    summary = {"splits": {}, "pairwise_overlap": {}}

    for split_name, indices in split_indices.items():
        frame = merged_df.iloc[indices]
        split_frames[split_name] = frame
        summary["splits"][split_name] = {
            "num_samples": int(len(frame)),
            "num_unique_drugs": int(frame["compound_iso_smiles"].nunique()),
            "num_unique_targets": int(frame["target_sequence"].nunique()),
            "num_unique_pairs": int(frame["pair_id"].nunique()),
            "num_unique_scaffolds": int(frame["scaffold_id"].nunique()),
        }

    pair_names = (("train", "val"), ("train", "test"), ("val", "test"))
    for left_name, right_name in pair_names:
        left_df = split_frames[left_name]
        right_df = split_frames[right_name]
        summary["pairwise_overlap"][f"{left_name}_{right_name}"] = {
            "shared_drugs": len(set(left_df["drug_id"]).intersection(set(right_df["drug_id"]))),
            "shared_targets": len(set(left_df["target_id"]).intersection(set(right_df["target_id"]))),
            "shared_pairs": len(set(left_df["pair_id"]).intersection(set(right_df["pair_id"]))),
            "shared_scaffolds": len(set(left_df["scaffold_id"]).intersection(set(right_df["scaffold_id"]))),
        }

    return summary
