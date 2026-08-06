# Running BioEmu Benchmarks

This guide describes the local workflow for evaluating generated structures and ensembles with the BioEmu benchmark scripts in this directory.

The workflow has two parts:

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

Run the MD-emulation evaluator from this directory:

```bash
python analyze_md_emulation.py
```

By default, the script reads:

```text
reference = md_emulation_benchmark_0.1/md_emulation/testcases.csv
sample_dir = samples
output_dir = results
```

To use other paths, edit the three variables in the `if __name__ == '__main__'` block of `analyze_md_emulation.py`:

```python
reference = '/PATH/TO/BIOEMU_ASSETS/md_emulation_benchmark_0.1/md_emulation/testcases.csv'
sample_dir = '/PATH/TO/SAMPLES'
results.save_results('/PATH/TO/RESULTS')
```

The script loads generated trajectories with `mdtraj`, projects them into the BioEmu MD-emulation coordinates, compares the sampled free-energy surface with the reference surface, and writes:

```text
RESULTS/
|-- results_metrics.csv
`-- results_projections.npz
```

It also prints aggregate `mae`, `rmse`, and `coverage` metrics.

## 3. Compare to Multi-Conformation References

Run the RMSD/contact comparison from this directory:

```bash
python compare_to_multi_conf.py /PATH/TO/PREDICTIONS
```

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

The per-case `*_contacts.npy` files store native-contact fractions for the local metric region.

`compare_to_multi_conf.py` currently filters to the `localunfolding` test cases:

```python
test_cases = ref_loca['test_case'].tolist()
```

To run a different subset, edit this line to use one or more of the loaded reference tables (`ref_cryp`, `ref_domi`, `ref_loca`, `ref_ood60`, `ref_oodval`).
