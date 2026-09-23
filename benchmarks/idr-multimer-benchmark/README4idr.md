# Running the IDR multimer DockQ benchmark

`get_dockq.py` reports the maximum DockQ between each predicted multimer ensemble and its reference structure. Paths are arguments. There is no default dataset location.

Run the script from this directory, or pass absolute paths. Scratch files are written to `./tmp` in the working directory.

## Install

The script needs `biotite`, `mdtraj`, `pandas`, `numpy`, and [DockQ](https://github.com/bjornwallner/DockQ).

## Directory layout

Predicted ensembles and reference structures share the same case name:

```text
TRAJ_DIR/
`-- CASE_NAME/
    |-- topology.pdb
    `-- traj_no_clash.xtc      # or traj.xtc, or traj.dcd

REF_DIR/
`-- CASE_NAME.pdb
```

Coordinates in `xtc` and `dcd` trajectories are treated as nanometers and converted to angstroms before DockQ. A case is skipped when `topology.pdb` or a supported trajectory file is missing, or when the reference PDB is missing.

Chain IDs do not have to be `A` and `B`. By default the script aligns sequences and renames the two matched chains. To force the reference mapping, pass a CSV:

```text
case,chain_keys,chain_ids
CASE_NAME,A:B,C:D
```

`chain_keys` are the labels used after renaming. `chain_ids` are the chain IDs in `CASE_NAME.pdb`, in the same order. An omitted CSV, or a case missing from the CSV, falls back to sequence matching.

## Run

```bash
python get_dockq.py \
    --traj_dir /PATH/TO/TRAJ_DIR \
    --ref_dir /PATH/TO/REF_DIR \
    --chain_id_dict /PATH/TO/multimer_chain_id.csv \
    --output /PATH/TO/dockq_scores.csv
```

`--chain_id_dict` is optional. `--output` defaults to `./dockq_scores.csv` in the working directory. The CSV index is the case name and the column is `dockq`.
