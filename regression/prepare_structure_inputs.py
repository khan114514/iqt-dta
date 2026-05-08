#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

import pandas as pd
import requests

try:  # Optional acceleration for residue-neighbor scoring.
    import numpy as np
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback keeps the script dependency-light
    np = None
    cKDTree = None


UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
DEFAULT_UNIPROT_FIELDS = (
    "accession,id,reviewed,protein_name,gene_names,organism_name,length,sequence,xref_pdb"
)
AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def command_version(command: Optional[str], args: Sequence[str]) -> Optional[str]:
    if not command:
        return None
    try:
        proc = subprocess.run(
            [command, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"version probe failed: {type(exc).__name__}: {exc}"
    return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""


def target_uid(sequence: str) -> str:
    return f"TGT_{sha256_text(sequence)[:16]}"


def load_dataset_targets(data_root: Path, datasets: Sequence[str]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    dataset_summary: Dict[str, object] = {}
    for dataset in datasets:
        raw_path = data_root / dataset / "raw" / "data.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw table: {raw_path}")
        frame = pd.read_csv(raw_path)
        required = {"compound_iso_smiles", "target_sequence", "affinity"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{raw_path} is missing required columns: {sorted(missing)}")
        grouped = (
            frame.reset_index()
            .groupby("target_sequence", sort=True)
            .agg(num_rows=("target_sequence", "size"), first_global_index=("index", "min"))
            .reset_index()
        )
        for _, row in grouped.iterrows():
            seq = str(row["target_sequence"])
            seq_hash = sha256_text(seq)
            rows.append(
                {
                    "dataset": dataset,
                    "target_uid": target_uid(seq),
                    "target_sequence_sha256": seq_hash,
                    "target_sequence": seq,
                    "sequence_length": len(seq),
                    "num_dataset_rows": int(row["num_rows"]),
                    "first_global_index": int(row["first_global_index"]),
                    "raw_table": str(raw_path),
                }
            )
        dataset_summary[dataset] = {
            "raw_table": str(raw_path),
            "raw_table_sha256": sha256_file(raw_path),
            "num_rows": int(len(frame)),
            "num_unique_targets": int(grouped.shape[0]),
            "num_unique_smiles": int(frame["compound_iso_smiles"].nunique()),
            "columns": list(map(str, frame.columns)),
        }
    return rows, dataset_summary


def download_uniprot_stream(
    cache_path: Path,
    query: str,
    fields: str,
    refresh: bool,
    timeout: int,
) -> Dict[str, object]:
    ensure_dir(cache_path.parent)
    params = {"query": query, "fields": fields, "format": "tsv", "compressed": "true"}
    request_url = f"{UNIPROT_STREAM_URL}?{urlencode(params)}"
    if cache_path.exists() and not refresh:
        return {
            "cache_path": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "retrieved_at_utc": None,
            "request_url": request_url,
            "source": "existing_cache",
        }

    response = requests.get(UNIPROT_STREAM_URL, params=params, timeout=timeout)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return {
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "retrieved_at_utc": utc_now(),
        "request_url": response.url,
        "source": "downloaded",
        "content_type": response.headers.get("content-type", ""),
        "content_encoding": response.headers.get("content-encoding", ""),
    }


def iter_uniprot_rows(cache_path: Path) -> Iterable[Dict[str, str]]:
    with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def is_reviewed(row: Dict[str, str]) -> bool:
    return str(row.get("Reviewed", "")).lower() == "reviewed"


def candidate_sort_key(row: Dict[str, str]) -> Tuple[int, str]:
    return (0 if is_reviewed(row) else 1, row.get("Entry", ""))


def build_uniprot_index(cache_path: Path, target_sequences: Sequence[str]) -> Tuple[Dict[str, List[Dict[str, str]]], int]:
    wanted = set(target_sequences)
    index: Dict[str, List[Dict[str, str]]] = {}
    total_rows = 0
    for row in iter_uniprot_rows(cache_path):
        total_rows += 1
        sequence = row.get("Sequence", "")
        if sequence in wanted:
            index.setdefault(sequence, []).append(row)
    for sequence in list(index):
        index[sequence] = sorted(index[sequence], key=candidate_sort_key)
    return index, total_rows


def clean_pdb_ids(value: object) -> str:
    if value is None:
        return ""
    parts = []
    for part in str(value).replace(",", ";").split(";"):
        token = part.strip()
        if token:
            parts.append(token)
    return ";".join(parts)


def mapping_rows(target_rows: Sequence[Dict[str, object]], uniprot_index: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for target in target_rows:
        sequence = str(target["target_sequence"])
        candidates = uniprot_index.get(sequence, [])
        selected = candidates[0] if candidates else {}
        if not candidates:
            status = "no_uniprot_exact_match"
        elif len(candidates) == 1:
            status = "exact_match"
        else:
            status = "ambiguous_exact_match_reviewed_preferred" if is_reviewed(selected) else "ambiguous_exact_match"
        rows.append(
            {
                **target,
                "mapping_status": status,
                "mapping_method": "UniProtKB exact full-sequence match",
                "candidate_count": len(candidates),
                "selected_uniprot_accession": selected.get("Entry", ""),
                "selected_uniprot_entry_name": selected.get("Entry Name", ""),
                "selected_uniprot_reviewed": selected.get("Reviewed", ""),
                "selected_gene_names": selected.get("Gene Names", ""),
                "selected_protein_names": selected.get("Protein names", ""),
                "selected_organism": selected.get("Organism", ""),
                "selected_uniprot_length": selected.get("Length", ""),
                "experimental_pdb_ids": clean_pdb_ids(selected.get("PDB", "")),
                "all_candidate_accessions": ";".join(row.get("Entry", "") for row in candidates),
                "all_candidate_reviewed": ";".join(row.get("Reviewed", "") for row in candidates),
            }
        )
    return rows


def request_json_with_retry(url: str, timeout: int, retries: int = 3) -> Tuple[Optional[object], Optional[str], Optional[int]]:
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 404:
                return None, "404 not found", response.status_code
            response.raise_for_status()
            return response.json(), None, response.status_code
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    return None, last_error, None


def alphafold_lookup(accession: str, timeout: int) -> Dict[str, object]:
    url = ALPHAFOLD_API_URL.format(accession=accession)
    if not accession:
        return {"uniprot_accession": accession, "alphafold_status": "not_queried", "alphafold_failure_reason": "missing UniProt accession"}
    payload, error, status_code = request_json_with_retry(url, timeout=timeout)
    if error:
        return {
            "uniprot_accession": accession,
            "alphafold_api_url": url,
            "alphafold_status": "missing",
            "alphafold_failure_reason": error,
            "alphafold_http_status": status_code or "",
        }
    if not payload:
        return {
            "uniprot_accession": accession,
            "alphafold_api_url": url,
            "alphafold_status": "missing",
            "alphafold_failure_reason": "empty API payload",
            "alphafold_http_status": status_code or "",
        }
    record = payload[0] if isinstance(payload, list) else payload
    return {
        "uniprot_accession": accession,
        "alphafold_api_url": url,
        "alphafold_status": "available",
        "alphafold_failure_reason": "",
        "alphafold_http_status": status_code or "",
        "alphafold_model_entity_id": record.get("modelEntityId", ""),
        "alphafold_latest_version": record.get("latestVersion", ""),
        "alphafold_tool_used": record.get("toolUsed", ""),
        "alphafold_model_created_date": record.get("modelCreatedDate", ""),
        "alphafold_sequence_version_date": record.get("sequenceVersionDate", ""),
        "alphafold_global_metric_value": record.get("globalMetricValue", ""),
        "alphafold_fraction_plddt_very_low": record.get("fractionPlddtVeryLow", ""),
        "alphafold_fraction_plddt_low": record.get("fractionPlddtLow", ""),
        "alphafold_fraction_plddt_confident": record.get("fractionPlddtConfident", ""),
        "alphafold_fraction_plddt_very_high": record.get("fractionPlddtVeryHigh", ""),
        "alphafold_sequence": record.get("sequence", ""),
        "alphafold_pdb_url": record.get("pdbUrl", ""),
        "alphafold_cif_url": record.get("cifUrl", ""),
        "alphafold_pae_doc_url": record.get("paeDocUrl", ""),
    }


def lookup_alphafold(accessions: Sequence[str], timeout: int, max_workers: int) -> Dict[str, Dict[str, object]]:
    unique_accessions = sorted({accession for accession in accessions if accession})
    results: Dict[str, Dict[str, object]] = {}
    if not unique_accessions:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(alphafold_lookup, accession, timeout): accession for accession in unique_accessions}
        for future in as_completed(futures):
            accession = futures[future]
            try:
                results[accession] = future.result()
            except Exception as exc:  # pragma: no cover - defensive diagnostic
                results[accession] = {
                    "uniprot_accession": accession,
                    "alphafold_status": "missing",
                    "alphafold_failure_reason": f"{type(exc).__name__}: {exc}",
                }
    return results


def download_file(url: str, output_path: Path, timeout: int) -> Tuple[bool, str]:
    if not url:
        return False, "missing URL"
    ensure_dir(output_path.parent)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        if output_path.suffix == ".gz":
            with gzip.open(output_path, "wb") as handle:
                handle.write(response.content)
        else:
            output_path.write_bytes(response.content)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def download_structures_for_manifest(
    manifest_rows: List[Dict[str, object]],
    structures_dir: Path,
    timeout: int,
    max_workers: int,
) -> None:
    download_jobs = []
    seen_accessions = set()
    for row in manifest_rows:
        accession = str(row.get("selected_uniprot_accession", ""))
        url = str(row.get("alphafold_pdb_url", ""))
        if row.get("structure_status") != "available" or not accession or not url or accession in seen_accessions:
            continue
        seen_accessions.add(accession)
        output_path = structures_dir / f"AF-{accession}-F1-model_v{row.get('alphafold_latest_version')}.pdb.gz"
        if output_path.exists():
            for manifest_row in manifest_rows:
                if str(manifest_row.get("selected_uniprot_accession", "")) == accession:
                    manifest_row["structure_file"] = str(output_path)
                    manifest_row["structure_file_sha256"] = sha256_file(output_path)
                    manifest_row["structure_cache_status"] = "downloaded"
            continue
        download_jobs.append((accession, url, output_path))

    if not download_jobs:
        return

    download_results: Dict[str, Tuple[bool, str, Path]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_file, url, output_path, timeout): (accession, output_path)
            for accession, url, output_path in download_jobs
        }
        for future in as_completed(futures):
            accession, output_path = futures[future]
            ok, error = future.result()
            download_results[accession] = (ok, error, output_path)

    for row in manifest_rows:
        accession = str(row.get("selected_uniprot_accession", ""))
        if accession not in download_results:
            continue
        ok, error, output_path = download_results[accession]
        if ok:
            row["structure_file"] = str(output_path)
            row["structure_file_sha256"] = sha256_file(output_path)
            row["structure_cache_status"] = "downloaded"
        else:
            row["structure_file"] = ""
            row["structure_file_sha256"] = ""
            row["structure_cache_status"] = "download_failed"
            row["structure_failure_reason"] = error


def build_structure_manifest(mapping: Sequence[Dict[str, object]], alphafold: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in mapping:
        accession = str(row.get("selected_uniprot_accession", ""))
        af = alphafold.get(accession, {})
        af_sequence = str(af.get("alphafold_sequence", ""))
        has_af = af.get("alphafold_status") == "available"
        if row["mapping_status"] == "no_uniprot_exact_match":
            status = "missing"
            reason = "no UniProt exact full-sequence match"
            source = ""
            structure_id = ""
        elif not has_af:
            status = "missing"
            reason = af.get("alphafold_failure_reason", "AlphaFold model unavailable")
            source = ""
            structure_id = ""
        elif af_sequence and af_sequence != row["target_sequence"]:
            status = "missing"
            reason = "AlphaFold API sequence does not match target sequence"
            source = ""
            structure_id = ""
        else:
            status = "available"
            reason = ""
            source = "AlphaFold DB"
            structure_id = str(af.get("alphafold_model_entity_id", ""))
        rows.append(
            {
                "dataset": row["dataset"],
                "target_uid": row["target_uid"],
                "target_sequence_sha256": row["target_sequence_sha256"],
                "sequence_length": row["sequence_length"],
                "mapping_status": row["mapping_status"],
                "selected_uniprot_accession": accession,
                "selected_uniprot_entry_name": row.get("selected_uniprot_entry_name", ""),
                "selected_uniprot_reviewed": row.get("selected_uniprot_reviewed", ""),
                "experimental_pdb_ids": row.get("experimental_pdb_ids", ""),
                "preferred_structure_source": source,
                "structure_id": structure_id,
                "structure_status": status,
                "structure_failure_reason": reason,
                "structure_file": "",
                "structure_file_sha256": "",
                "structure_cache_status": "metadata_only" if status == "available" else "missing",
                "alphafold_api_url": af.get("alphafold_api_url", ""),
                "alphafold_pdb_url": af.get("alphafold_pdb_url", ""),
                "alphafold_cif_url": af.get("alphafold_cif_url", ""),
                "alphafold_pae_doc_url": af.get("alphafold_pae_doc_url", ""),
                "alphafold_latest_version": af.get("alphafold_latest_version", ""),
                "alphafold_tool_used": af.get("alphafold_tool_used", ""),
                "alphafold_model_created_date": af.get("alphafold_model_created_date", ""),
                "alphafold_sequence_version_date": af.get("alphafold_sequence_version_date", ""),
                "alphafold_global_metric_value": af.get("alphafold_global_metric_value", ""),
                "alphafold_sequence_matches_target": bool(af_sequence and af_sequence == row["target_sequence"]),
            }
        )
    return rows


def open_text_maybe_gzip(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
    else:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            yield from handle


def parse_pdb_residues(path: Path) -> List[Dict[str, object]]:
    residue_order: List[Tuple[str, str, str, str]] = []
    residues: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for line in open_text_maybe_gzip(path):
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        altloc = line[16].strip()
        if altloc not in {"", "A"}:
            continue
        resname = line[17:20].strip()
        chain = line[21].strip() or "_"
        resseq = line[22:26].strip()
        icode = line[26].strip()
        key = (chain, resseq, icode, resname)
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            bfactor = float(line[60:66])
        except ValueError:
            continue
        if key not in residues:
            residues[key] = {
                "chain": chain,
                "residue_number": resseq,
                "insertion_code": icode,
                "residue_name": resname,
                "aa": AA3_TO_AA1.get(resname, "X"),
                "atoms": [],
                "bfactors": [],
                "ca": None,
            }
            residue_order.append(key)
        residues[key]["atoms"].append((x, y, z))
        residues[key]["bfactors"].append(bfactor)
        if atom_name == "CA":
            residues[key]["ca"] = (x, y, z)

    parsed = []
    for index, key in enumerate(residue_order, start=1):
        residue = residues[key]
        coords = residue["atoms"]
        ca = residue["ca"]
        if ca is None and coords:
            ca = tuple(sum(coord[i] for coord in coords) / len(coords) for i in range(3))
        if ca is None:
            continue
        residue["residue_index"] = index
        residue["ca"] = ca
        residue["mean_plddt"] = sum(residue["bfactors"]) / len(residue["bfactors"]) if residue["bfactors"] else ""
        parsed.append(residue)
    return parsed


def euclidean(left: Tuple[float, float, float], right: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((left[i] - right[i]) ** 2 for i in range(3)))


def heuristic_pocket_scores(structure_file: Path) -> List[Dict[str, object]]:
    residues = parse_pdb_residues(structure_file)
    coords = [residue["ca"] for residue in residues]
    if np is not None and cKDTree is not None and coords:
        coord_array = np.asarray(coords, dtype=float)
        tree = cKDTree(coord_array)
        neighbors_8_values = [len(indices) - 1 for indices in tree.query_ball_point(coord_array, r=8.0)]
        neighbors_12_values = [len(indices) - 1 for indices in tree.query_ball_point(coord_array, r=12.0)]
    else:
        neighbors_8_values = []
        neighbors_12_values = []
        for ca in coords:
            distances = [euclidean(ca, other) for other in coords]
            neighbors_8_values.append(sum(1 for distance in distances if 0.0 < distance <= 8.0))
            neighbors_12_values.append(sum(1 for distance in distances if 0.0 < distance <= 12.0))

    rows: List[Dict[str, object]] = []
    for idx, residue in enumerate(residues):
        ca = residue["ca"]
        neighbors_8 = neighbors_8_values[idx]
        neighbors_12 = neighbors_12_values[idx]
        local_density = min(neighbors_12 / 30.0, 1.0)
        moderate_exposure = 1.0 - min(abs(neighbors_8 - 10.0) / 16.0, 1.0)
        confidence = float(residue["mean_plddt"]) / 100.0 if residue["mean_plddt"] != "" else 0.0
        score = confidence * (0.65 * local_density + 0.35 * moderate_exposure)
        rows.append(
            {
                "residue_index": residue["residue_index"],
                "chain": residue["chain"],
                "residue_number": residue["residue_number"],
                "insertion_code": residue["insertion_code"],
                "residue_name": residue["residue_name"],
                "aa": residue["aa"],
                "ca_x": f"{ca[0]:.3f}",
                "ca_y": f"{ca[1]:.3f}",
                "ca_z": f"{ca[2]:.3f}",
                "mean_plddt": f"{float(residue['mean_plddt']):.3f}" if residue["mean_plddt"] != "" else "",
                "neighbors_8a": neighbors_8,
                "neighbors_12a": neighbors_12,
                "local_density_score": f"{local_density:.6f}",
                "moderate_exposure_score": f"{moderate_exposure:.6f}",
                "heuristic_pocket_score": f"{score:.6f}",
            }
        )
    return rows


def write_pocket_scores(
    manifest_rows: Sequence[Dict[str, object]],
    output_dir: Path,
    method: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, object]] = []
    fieldnames = [
        "residue_index",
        "chain",
        "residue_number",
        "insertion_code",
        "residue_name",
        "aa",
        "ca_x",
        "ca_y",
        "ca_z",
        "mean_plddt",
        "neighbors_8a",
        "neighbors_12a",
        "local_density_score",
        "moderate_exposure_score",
        "heuristic_pocket_score",
    ]
    score_cache: Dict[str, List[Dict[str, object]]] = {}
    for row in manifest_rows:
        dataset = str(row["dataset"])
        target = str(row["target_uid"])
        score_file = output_dir / "pocket_scores" / dataset / f"{target}.csv"
        structure_file = str(row.get("structure_file", ""))
        base = {
            "dataset": dataset,
            "target_uid": target,
            "target_sequence_sha256": row.get("target_sequence_sha256", ""),
            "selected_uniprot_accession": row.get("selected_uniprot_accession", ""),
            "pocket_method": method,
            "structure_file": structure_file,
            "pocket_score_file": str(score_file),
        }
        if method == "none":
            failure = "pocket scoring disabled"
            summary_rows.append({**base, "pocket_status": "not_generated", "num_residue_scores": 0, "failure_reason": failure})
            failed_rows.append({**base, "failure_stage": "pocket", "failure_reason": failure})
            continue
        if row.get("structure_cache_status") != "downloaded" or not structure_file:
            failure = row.get("structure_failure_reason") or "structure file not downloaded"
            summary_rows.append({**base, "pocket_status": "not_generated", "num_residue_scores": 0, "failure_reason": failure})
            failed_rows.append({**base, "failure_stage": "pocket", "failure_reason": failure})
            continue
        if method != "heuristic":
            failure = f"unsupported pocket method: {method}"
            summary_rows.append({**base, "pocket_status": "not_generated", "num_residue_scores": 0, "failure_reason": failure})
            failed_rows.append({**base, "failure_stage": "pocket", "failure_reason": failure})
            continue
        try:
            if structure_file not in score_cache:
                score_cache[structure_file] = heuristic_pocket_scores(Path(structure_file))
            score_rows = score_cache[structure_file]
            write_csv(score_file, score_rows, fieldnames)
            summary_rows.append({**base, "pocket_status": "generated", "num_residue_scores": len(score_rows), "failure_reason": ""})
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            summary_rows.append({**base, "pocket_status": "failed", "num_residue_scores": 0, "failure_reason": failure})
            failed_rows.append({**base, "failure_stage": "pocket", "failure_reason": failure})
    return summary_rows, failed_rows


def build_failed_rows(
    mapping: Sequence[Dict[str, object]],
    manifest: Sequence[Dict[str, object]],
    pocket_failures: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in mapping:
        if row["mapping_status"] == "no_uniprot_exact_match":
            rows.append(
                {
                    "dataset": row["dataset"],
                    "target_uid": row["target_uid"],
                    "target_sequence_sha256": row["target_sequence_sha256"],
                    "selected_uniprot_accession": "",
                    "failure_stage": "mapping",
                    "failure_reason": "no UniProt exact full-sequence match",
                }
            )
    for row in manifest:
        if row["structure_status"] != "available" or row["structure_cache_status"] not in {"metadata_only", "downloaded"}:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "target_uid": row["target_uid"],
                    "target_sequence_sha256": row["target_sequence_sha256"],
                    "selected_uniprot_accession": row.get("selected_uniprot_accession", ""),
                    "failure_stage": "structure",
                    "failure_reason": row.get("structure_failure_reason", "structure unavailable"),
                }
            )
    rows.extend(dict(row) for row in pocket_failures)
    return rows


def coverage_summary(
    target_rows: Sequence[Dict[str, object]],
    mapping: Sequence[Dict[str, object]],
    manifest: Sequence[Dict[str, object]],
    pockets: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    datasets = sorted({row["dataset"] for row in target_rows})
    for dataset in datasets:
        targets = [row for row in target_rows if row["dataset"] == dataset]
        mapped = [row for row in mapping if row["dataset"] == dataset and row["mapping_status"] != "no_uniprot_exact_match"]
        exact = [row for row in mapping if row["dataset"] == dataset and row["mapping_status"] == "exact_match"]
        ambiguous = [row for row in mapping if row["dataset"] == dataset and str(row["mapping_status"]).startswith("ambiguous")]
        reviewed = [row for row in mapping if row["dataset"] == dataset and row.get("selected_uniprot_reviewed") == "reviewed"]
        structure = [row for row in manifest if row["dataset"] == dataset and row["structure_status"] == "available"]
        downloaded = [row for row in manifest if row["dataset"] == dataset and row["structure_cache_status"] == "downloaded"]
        pocket_generated = [row for row in pockets if row["dataset"] == dataset and row["pocket_status"] == "generated"]
        total = len(targets)
        summary[dataset] = {
            "num_targets": total,
            "mapped_targets": len(mapped),
            "mapped_coverage": len(mapped) / total if total else 0.0,
            "unambiguous_exact_matches": len(exact),
            "ambiguous_exact_matches": len(ambiguous),
            "reviewed_selected_targets": len(reviewed),
            "alphafold_available_targets": len(structure),
            "alphafold_available_coverage": len(structure) / total if total else 0.0,
            "structure_files_downloaded": len(downloaded),
            "pocket_score_files_generated": len(pocket_generated),
        }
    combined_sequences = {row["target_sequence_sha256"] for row in target_rows}
    mapped_sequences = {
        row["target_sequence_sha256"]
        for row in mapping
        if row["mapping_status"] != "no_uniprot_exact_match"
    }
    structure_sequences = {
        row["target_sequence_sha256"]
        for row in manifest
        if row["structure_status"] == "available"
    }
    summary["combined_unique_sequences"] = {
        "num_targets": len(combined_sequences),
        "mapped_targets": len(mapped_sequences),
        "mapped_coverage": len(mapped_sequences) / len(combined_sequences) if combined_sequences else 0.0,
        "alphafold_available_targets": len(structure_sequences),
        "alphafold_available_coverage": len(structure_sequences) / len(combined_sequences) if combined_sequences else 0.0,
    }
    return summary


def write_reproduction_commands(path: Path, args: argparse.Namespace) -> None:
    ensure_dir(path.parent)
    datasets = " ".join(args.datasets)
    download_flag = " --download_structures" if args.download_structures else ""
    refresh_flag = " --refresh_uniprot" if args.refresh_uniprot else ""
    command = (
        '${PYTHON_BIN:-./.conda/bin/python} regression/prepare_structure_inputs.py '
        f'--data_root "{args.data_root}" '
        f'--output_dir "{args.output_dir}" '
        f"--datasets {datasets} "
        f'--uniprot_query "{args.uniprot_query}" '
        f"--pocket_method {args.pocket_method} "
        f"--max_workers {args.max_workers}"
        f"{download_flag}{refresh_flag}"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")
        handle.write("# Rebuild Davis/KIBA optional structure-annotation audit artifacts.\n")
        handle.write(command)
        handle.write("\n")
    path.chmod(0o755)


def write_dynamic_readme(path: Path, summary: Dict[str, object], args: argparse.Namespace) -> None:
    lines = [
        "# Optional Structure Annotation Audit Bundle",
        "",
        "This directory records how Davis/KIBA target sequences are mapped to external protein structure resources for optional annotation and audit.",
        "The current model code in this repository still consumes sequence tensors; no training or inference path reads the AlphaFold/PDB files or pocket-score CSV files.",
        "",
        "Generated files:",
        "",
        "- `target_sequences.csv`: unique target sequences extracted from the released raw Davis/KIBA tables.",
        "- `target_mapping.csv`: exact full-sequence UniProtKB mapping table with candidate counts and selected accessions.",
        "- `structure_cache_manifest.csv`: AlphaFold DB structure availability, URLs, optional local PDB cache paths, and experimental PDB cross-references from UniProt.",
        "- `pocket_scores_summary.csv`: one row per dataset target describing whether a pocket score file was generated.",
        "- `pocket_scores/<dataset>/<target_uid>.csv`: residue-level heuristic structure scores when `--pocket_method heuristic --download_structures` is used.",
        "- `failed_targets.csv`: mapping, structure, or pocket-generation failures with explicit reasons.",
        "- `versions.json`: API endpoints, cache hashes, local tool probes, and coverage summary.",
        "- `commands.sh`: exact command used to rebuild the bundle.",
        "",
        "Pocket method:",
        "",
        f"- Selected method: `{args.pocket_method}`.",
        "- `heuristic` is a deterministic residue-level local-density/confidence score computed from AlphaFold PDB coordinates and pLDDT B-factors. It is not P2Rank or fpocket.",
        "- If the manuscript claims a structure-aware model branch or P2Rank/fpocket pockets, the model/data pipeline and this bundle must be updated accordingly.",
        "- As released here, the structural files are optional annotation/audit artifacts rather than model inputs.",
        "",
        "Coverage summary:",
        "",
        "| dataset | targets | mapped | AlphaFold available | structures downloaded | pocket files |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(k for k in summary if k != "combined_unique_sequences"):
        item = summary[dataset]
        lines.append(
            f"| {dataset} | {item['num_targets']} | {item['mapped_targets']} | "
            f"{item['alphafold_available_targets']} | {item['structure_files_downloaded']} | "
            f"{item['pocket_score_files_generated']} |"
        )
    combined = summary.get("combined_unique_sequences", {})
    if combined:
        lines.extend(
            [
                "",
                "Combined unique target sequences:",
                "",
                f"- total: {combined.get('num_targets')}",
                f"- mapped: {combined.get('mapped_targets')}",
                f"- AlphaFold available: {combined.get('alphafold_available_targets')}",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def fieldnames_for(rows: Sequence[Dict[str, object]]) -> List[str]:
    names: List[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare reproducible target-structure mapping artifacts for Davis/KIBA.")
    parser.add_argument("--data_root", default="regression/data", help="Root directory containing {dataset}/raw/data.csv.")
    parser.add_argument("--output_dir", default="regression/data/structure_inputs", help="Output bundle directory.")
    parser.add_argument("--datasets", nargs="+", default=["davis", "kiba"], help="Dataset names to audit.")
    parser.add_argument("--uniprot_query", default="(organism_id:9606)", help="UniProtKB search query used for exact sequence matching.")
    parser.add_argument("--uniprot_fields", default=DEFAULT_UNIPROT_FIELDS, help="Comma-separated UniProtKB fields to cache.")
    parser.add_argument("--refresh_uniprot", action="store_true", help="Refresh cached UniProt stream TSV.")
    parser.add_argument("--download_structures", action="store_true", help="Download AlphaFold PDB files into output_dir/structures.")
    parser.add_argument("--pocket_method", choices=["none", "heuristic"], default="none", help="Pocket score generation method.")
    parser.add_argument("--max_workers", type=int, default=8, help="Concurrent workers for AlphaFold lookup/download.")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds.")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    target_rows, dataset_summary = load_dataset_targets(data_root, args.datasets)
    unique_sequences = sorted({str(row["target_sequence"]) for row in target_rows})
    uniprot_cache_path = output_dir / "cache" / "uniprotkb_target_search.tsv.gz"
    uniprot_cache = download_uniprot_stream(
        cache_path=uniprot_cache_path,
        query=args.uniprot_query,
        fields=args.uniprot_fields,
        refresh=args.refresh_uniprot,
        timeout=args.timeout,
    )
    uniprot_index, uniprot_total_rows = build_uniprot_index(uniprot_cache_path, unique_sequences)
    mapping = mapping_rows(target_rows, uniprot_index)
    alphafold = lookup_alphafold(
        [str(row.get("selected_uniprot_accession", "")) for row in mapping],
        timeout=args.timeout,
        max_workers=max(1, args.max_workers),
    )
    manifest = build_structure_manifest(mapping, alphafold)
    if args.download_structures:
        download_structures_for_manifest(
            manifest,
            structures_dir=output_dir / "structures" / "alphafold",
            timeout=args.timeout,
            max_workers=max(1, args.max_workers),
        )
    pocket_summary, pocket_failures = write_pocket_scores(manifest, output_dir, args.pocket_method)
    failures = build_failed_rows(mapping, manifest, pocket_failures)
    summary = coverage_summary(target_rows, mapping, manifest, pocket_summary)

    write_csv(output_dir / "target_sequences.csv", target_rows, fieldnames_for(target_rows))
    write_csv(output_dir / "target_mapping.csv", mapping, fieldnames_for(mapping))
    write_csv(output_dir / "structure_cache_manifest.csv", manifest, fieldnames_for(manifest))
    write_csv(output_dir / "pocket_scores_summary.csv", pocket_summary, fieldnames_for(pocket_summary))
    write_csv(output_dir / "failed_targets.csv", failures, fieldnames_for(failures) if failures else ["dataset", "target_uid", "target_sequence_sha256", "selected_uniprot_accession", "failure_stage", "failure_reason"])

    fpocket = shutil.which("fpocket")
    prank = shutil.which("prank") or shutil.which("p2rank")
    versions = {
        "created_at_utc": utc_now(),
        "command": " ".join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "requests_version": requests.__version__,
        "uniprot": {
            "stream_url": UNIPROT_STREAM_URL,
            "query": args.uniprot_query,
            "fields": args.uniprot_fields,
            "cache": uniprot_cache,
            "total_rows_scanned": uniprot_total_rows,
        },
        "alphafold": {
            "api_url_template": ALPHAFOLD_API_URL,
            "queried_accessions": len(alphafold),
        },
        "pocket_scoring": {
            "method": args.pocket_method,
            "fpocket_path": fpocket or "",
            "fpocket_version_probe": command_version(fpocket, ["--version"]) if fpocket else "",
            "p2rank_path": prank or "",
            "p2rank_version_probe": command_version(prank, ["--version"]) if prank else "",
        },
        "datasets": dataset_summary,
        "coverage_summary": summary,
    }
    write_json(output_dir / "versions.json", versions)
    write_reproduction_commands(output_dir / "commands.sh", args)
    write_dynamic_readme(output_dir / "README.md", summary, args)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
