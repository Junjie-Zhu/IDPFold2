# Running BioEmu Benchmarks

This guide describes the local workflow for evaluating generated structures and ensembles with the BioEmu benchmark scripts in this directory.

The workflow has three parts:

- TICA fit and projection with `fit_tica.py` and `project_tica.py`.
- MD-emulation free-energy metrics with `analyze_md_emulation.py`.
- Multi-conformation RMSD/contact comparison with `compare_to_multi_conf.py`.

Use absolute paths in the commands below, or replace every `/PATH/TO/...` placeholder consistently before running. Run the commands from this directory unless stated otherwise.

## Directory and Input Conventions

The example commands use these placeholders:

- `/PATH/TO/IDPFold2`: clone of this repository.
- `/PATH/TO/BIOEMU_ASSETS`: BioEmu benchmark assets copied into this directory.
- `/PATH/TO/SAMPLES`: generated MD-emulation samples.
- `/PATH/TO/PREDICTIONS`: generated single- or multi-model PDB predictions.
- `/PATH/TO/RESULTS`: output directory for MD-emulation metrics.

Download the BioEmu benchmark data from [microsoft/bioemu-benchmarks](https://github.com/microsoft/bioemu-benchmarks). The files used here are under `bioemu-benchmarks/assets/`.

After downloading, place or link the asset directories next to these scripts:

```text
./
|-- analyze_md_emulation.py
|-- compare_to_multi_conf.py
|-- fit_tica.py
|-- project_tica.py
|-- projection.py
|-- state_metric.py
|-- utils.py
|-- md_emulation_benchmark_0.1/
|-- crypticpocket/
|-- domainmotion/
|-- localunfolding/
|-- ood60/
`-- oodval/
```

For MD emulation, each sample should be stored by BioEmu test-case ID. The parallel loader in `analyze_md_emulation.py` expects this layout:

```text
SAMPLES/
|-- TEST_CASE_A/
|   |-- topology.pdb
|   `-- traj.dcd
`-- TEST_CASE_B/
    |-- topology.pdb
    `-- traj.dcd
```

For multi-conformation comparison, put predicted PDB files directly under `/PATH/TO/PREDICTIONS`. File names should contain BioEmu test-case IDs. If one prediction file contains several IDs separated by `:`, the script copies the file for the first matching benchmark case.

## 1. Prepare BioEmu Assets

Copy or link the upstream assets into this directory. The MD-emulation workflow uses:

```text
md_emulation_benchmark_0.1/md_emulation/testcases.csv
md_emulation_benchmark_0.1/md_emulation/reference_projections.npz
md_emulation_benchmark_0.1/md_emulation/projections_sqrt_inv_cov.npz
md_emulation_benchmark_0.1/md_emulation/projections_mean.npz
```

The multi-conformation workflow uses the `references.csv`, `reference/<test_case>/`, and optional `local_residinfo/<test_case>.json` files inside these benchmark folders:

```text
crypticpocket/
domainmotion/
localunfolding/
ood60/
oodval/
```

## 2. Analyze MD Emulation

Run the MD-emulation evaluator from this directory. Reference assets are read from the directory that contains the script, so the default `--reference` still finds `testcases.csv` when the assets sit next to these files. `--sample_dir` and `--output_dir` are relative to the working directory.

```bash
python analyze_md_emulation.py \
    --reference /PATH/TO/BIOEMU_ASSETS/md_emulation_benchmark_0.1/md_emulation/testcases.csv \
    --sample_dir /PATH/TO/SAMPLES \
    --output_dir /PATH/TO/RESULTS
```

The defaults are:

```text
--reference   <this directory>/md_emulation_benchmark_0.1/md_emulation/testcases.csv
--sample_dir  samples
--output_dir  results
```

`testcases.csv` must contain a `test_case` column. Every listed case is loaded from `{sample_dir}/{test_case}/topology.pdb` and `{sample_dir}/{test_case}/traj.dcd`.

The script loads generated trajectories with `mdtraj`, projects them into the BioEmu MD-emulation coordinates, compares the sampled free-energy surface with the reference surface, and writes:

```text
RESULTS/
|-- results_metrics.csv
`-- results_projections.npz
```

It also prints aggregate `mae`, `rmse`, and `coverage` metrics.

## 3. Compare to Multi-Conformation References

Reference tables and PDB files are read from the benchmark folders next to `compare_to_multi_conf.py`, not from the working directory.

```bash
python compare_to_multi_conf.py /PATH/TO/PREDICTIONS --benchmark localunfolding
```

`--benchmark` selects the reference table. The choices are `localunfolding` (the default), `domainmotion`, `crypticpocket`, `ood60`, and `oodval`. Figure 2 uses `localunfolding` for the native-contact fraction and `domainmotion` for the local RMSD.

The script creates a processing directory under `/PATH/TO/PREDICTIONS`:

```text
PREDICTIONS/
|-- processing/
|   |-- TEST_CASE_A.pdb
|   |-- TEST_CASE_A_contacts.npy
|   `-- ...
`-- metrics_rmsd.pkl
```

`metrics_rmsd.pkl` contains a pickled dictionary with:

- `test_case`: BioEmu test-case IDs.
- `ref`: reference PDB files used for each case.
- `local_rmsd`: local-region RMSD values.
- `global_rmsd`: full matched-structure RMSD values.

The per-case `*_contacts.npy` files store native-contact fractions for the local metric region. A native contact is a Cα pair under 8 Å with sequence separation of at least 3 residues in the reference; the fraction is how many of those pairs are recovered in each model.

## 4. Fast-folding TICA

This path is separate from MD emulation. MD emulation projects Cα contacts with the matrices shipped in the BioEmu assets. Fast-folding TICA fits a deeptime model on all Cα–Cα distances of a reference trajectory, then projects other ensembles into that model. The two `projected_data` arrays are not interchangeable.

Install `deeptime` in addition to the packages in `requirements.txt`. Features are mdtraj distances in nanometers. The sample must have the same Cα count and order as the reference.

Fit one reference PDB, one topology-plus-trajectory, or a directory of either:

```bash
python fit_tica.py /PATH/TO/REFERENCE.pdb --output-dir /PATH/TO/TICA

python fit_tica.py /PATH/TO/TRAJ.xtc \
    --topology /PATH/TO/topology.pdb \
    --output-dir /PATH/TO/TICA \
    --lag-time 10 \
    --dim 5
```

A directory may contain `{name}.pdb` files or subdirectories `{name}/topology.pdb` plus `traj.xtc`, `traj.dcd`, `samples.xtc`, or `traj_no_clash.xtc`. Each fit is `{name}_tica.npz` with the pickled model and the reference projection in `projected_data`. The default lag is 10 frames and the default dimension is 5. The trajectory needs more frames than the lag, and `dim` must be at least 2.

Project the predicted ensemble, and optionally a BioEmu ensemble, with those models:

```bash
python project_tica.py \
    --model-dir /PATH/TO/TICA \
    --sample-dir /PATH/TO/SAMPLES \
    --bioemu-dir /PATH/TO/BIOEMU \
    --output-dir /PATH/TO/RESULTS
```

`--sample-dir` and `--bioemu-dir` use the same layout as the fitter: `{name}.pdb` or `{name}/topology.pdb` plus a trajectory. `--bioemu-dir` can be omitted.

`/PATH/TO/RESULTS` then contains:

```text
RESULTS/
|-- {name}_ref.npz
|-- {name}_pred.npz
|-- {name}_bioemu.npz                  # only with --bioemu-dir
|-- results_metrics.csv
`-- results_bioemu_metrics.csv         # only with --bioemu-dir
```

Each npz file has one array, `projected_data`. `results_metrics.csv` compares the sample with the reference on the first two TICA components. Columns are `test_case`, `mae`, `rmse`, and `coverage`, and the last row is `mean`. `mae` is the shift-minimized free-energy mean absolute error in kcal/mol. `results_bioemu_metrics.csv` is the same comparison for the BioEmu trajectories.
