# IQT-DTA

Minimal code release for the sequence-based MGraphDTA regression pipeline and
the optional Davis/KIBA structure-annotation audit workflow.

This repository intentionally does not include datasets, processed `.pt` files,
checkpoints, experiment outputs, AlphaFold/PDB caches, or pocket-score CSVs.
Provide your own Davis/KIBA-style data under `regression/data/<dataset>/raw/`
when running the scripts.

Expected raw data files:

```text
regression/data/davis/raw/data.csv
regression/data/davis/raw/data_train.csv
regression/data/davis/raw/data_test.csv
regression/data/kiba/raw/data.csv
regression/data/kiba/raw/data_train.csv
regression/data/kiba/raw/data_test.csv
```

Each CSV should contain:

```text
compound_iso_smiles,target_sequence,affinity
```

Install dependencies in an environment with PyTorch, PyTorch Geometric, RDKit,
NumPy, pandas, scikit-learn, SciPy, requests, NetworkX, and tqdm.

Preprocess a dataset:

```bash
python regression/preprocessing.py --data_root regression/data --datasets davis kiba
```

Train:

```bash
python regression/train.py \
  --dataset davis \
  --split_type random \
  --seed 0 \
  --variant full_model \
  --data_root regression/data \
  --results_root results
```

Evaluate:

```bash
python regression/test.py \
  --run_dir results/davis/MGraphDTA/full_model/random/seed_0/<run_name>
```

Build the optional structure-annotation audit bundle from user-provided raw
Davis/KIBA tables:

```bash
python regression/prepare_structure_inputs.py \
  --data_root regression/data \
  --output_dir regression/data/structure_inputs \
  --datasets davis kiba \
  --uniprot_query "(organism_id:9606)" \
  --download_structures \
  --pocket_method heuristic
```

The structure-annotation workflow maps target sequences to UniProtKB by exact
full-sequence match, records AlphaFold DB availability and UniProt PDB
cross-references, optionally caches AlphaFold PDB files, and can generate
heuristic residue-level pocket annotation CSVs. These files are audit artifacts:
the released model code consumes ligand molecular graphs and protein sequence
tensors, not receptor 3D pocket graphs.
