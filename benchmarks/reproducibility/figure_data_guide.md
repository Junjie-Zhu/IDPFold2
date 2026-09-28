# From structures to figure data

This guide maps each panel in Figures 2–4 to the quantity it draws and the command that produces that quantity.

Run every command from the IDPFold2 repository root. Paths below are relative to that root. Replace every `/PATH/TO/...` placeholder before running. Output names describe the fields a panel reads. They do not have to match the historical `data_figure_*` filenames.

**Notes**

* Histograms, free-energy surfaces (`-kT ln P` at 300 K), LOWESS curves, AUC, linear fits, ROC/PRC, and t-tests are computed inside the plotting scripts.
* Lengths below are in Å unless a plot axis is marked in nm, in which case the plot divides by 10.
* G-scores are a PeptoneDB column. They are not computed from the ensemble.

## Commands

| Quantity | Command |
| --- | --- |
| Rg, end-to-end distance, DSSP helix/coil/disorder, Cα distance maps | `python scripts/quick_analysis.py /PATH/TO/ENSEMBLE [--reference-dir /PATH/TO/REFS]` |
| Case I(q) table and predicted SAXS `.dat` | `python benchmarks/reproducibility/prepare_saxs_cases.py --experimental-dir /PATH/TO/DAT --pepsi-dir /PATH/TO/PEPSI --output-dir /PATH/TO/OUT` |
| Pair distance distribution P(r) | `python benchmarks/reproducibility/calculate_pr.py /PATH/TO/DAT_DIR --output-dir /PATH/TO/OUT --angular-unit 1/A` |
| Fast-folding TICA | `python benchmarks/bioemu-benchmark/fit_tica.py` then `python benchmarks/bioemu-benchmark/project_tica.py` ([README4bioemu.md](../bioemu-benchmark/README4bioemu.md), section 4) |
| MD-emulation free-energy landscapes, MAE, coverage | `python benchmarks/bioemu-benchmark/analyze_md_emulation.py` ([README4bioemu.md](../bioemu-benchmark/README4bioemu.md), section 2) |
| Native-contact fraction and local RMSD | `python benchmarks/bioemu-benchmark/compare_to_multi_conf.py /PATH/TO/PREDICTIONS --benchmark localunfolding` |
| DockQ | `python benchmarks/idr-multimer-benchmark/get_dockq.py` ([README4idr.md](../idr-multimer-benchmark/README4idr.md)) |
| SAXS I(q) forward model, chemical shifts, PRE, RDC, reweighted errors | [README4peptone.md](../peptonebench/README4peptone.md) |

Fast-folding TICA and MD emulation are different projections. TICA is a deeptime model fit on all Cα–Cα distances. MD emulation projects Cα contacts with the matrices shipped in the BioEmu assets. The two `projected_data` arrays are not interchangeable.

## `quick_analysis.py`

`python scripts/quick_analysis.py /PATH/TO/ENSEMBLE` reads a directory of multi-model PDB files (`{name}.pdb`). It writes `metrics.pkl` and one `{name}_contact_map.npy` per system into that directory. DSSP needs backbone atoms (N, CA, C, O) and an `mkdssp` binary on `PATH`. Coarse-grained ensembles have to be backmapped with `scripts/_cg2all.py` before DSSP. If DSSP fails, Rg, end-to-end distance, and the distance map are still written.

`metrics.pkl` is a dictionary of lists, one entry per system:

- `name`
- `rg_predict`: per-model radius of gyration (biotite), Å
- `re2e_predict`: per-model Cα distance from the first residue to the last, Å
- `helix_content`, `coil_content`: per-residue fraction of models whose simplified DSSP code is `H` or `C`
- `disorder_ratio`: mean, over models, of the fraction of residues with code `C`
- `full_helix_content`, `full_coil_content`: same fractions. Without a peptide-to-complex mapping they copy the prediction. Mapping runs only when the prediction is one chain and the reference PDB has several chains: `helix_content` / `coil_content` are then the reference chain, and `full_*` are the aligned peptide residues

`{name}_contact_map.npy` is a dictionary:

- `pred`: ensemble-mean Cα–Cα distance matrix, Å, shape `(N, N)`
- `all_pred`: per-model distances, shape `(n_models, N, N)`
- `chains`: residue counts per chain, in PDB order
- `ref`: Cα–Cα matrix of `{reference-dir}/{name}.pdb`, present only when that file exists

## SAXS case curves and P(r)

`prepare_saxs_cases.py` pairs each `Pepsi-{name}.csv` with `{name}.dat` or `{name}-bift.dat`.

```bash
python benchmarks/reproducibility/prepare_saxs_cases.py \
    --experimental-dir /PATH/TO/DAT \
    --pepsi-dir /PATH/TO/PEPSI \
    --output-dir /PATH/TO/SAXS_CASES
```

`/PATH/TO/SAXS_CASES` then contains:

- `{name}_intensity.csv`: `q`, `Iq_exp`, `sigma`, `Iq_pred`. `q` is the experimental grid times `--q-scale` (default 10, so 1/Å becomes 1/nm). `Iq_pred` is the ensemble-mean Pepsi intensity interpolated onto that grid.
- `{name}_pred.dat`: `q`, ensemble-mean `I(q)`, and a per-q sigma, on the Pepsi q grid and in the Pepsi q unit.

Run P(r) once on the experimental profiles and once on the predicted profiles:

```bash
python benchmarks/reproducibility/calculate_pr.py /PATH/TO/DAT \
    --output-dir /PATH/TO/PR_EXP --angular-unit 1/A

python benchmarks/reproducibility/calculate_pr.py /PATH/TO/SAXS_CASES \
    --output-dir /PATH/TO/PR_PRED --angular-unit 1/A
```

The second command only reads `*.dat`. The intensity CSV in the same directory is ignored. Each profile writes `{sample}_pr_fits.npy` (every fit) and `{sample}_best_pr_fit.csv` (the selected fit). `{sample}` is the `.dat` filename. The selected row's `Index` picks one entry of the `.npy` dictionary. That entry's `p(r)` is a length-2 array of r and P(r). The row's `G Rg` and `Dmax` are the Rg and Dmax of that fit. A P(r) panel reads those four values from the experimental fit and from the predicted fit (`dist_exp`, `pr_exp`, `pr_rg_exp`, `dmax_exp`, and the matching `*_pred` fields).

## Figure 2

### Panels a–d

Ensemble-average Rg, and the end-to-end distance distribution.

```bash
python scripts/quick_analysis.py /PATH/TO/ENSEMBLE
```

Panels read `rg_predict` and `re2e_predict` from `metrics.pkl`. Experimental Rg, where a panel compares against it, is an external table joined on the system name.

### Panels e, f

Free-energy landscapes in the top two collective variables, plus MAE and coverage against the reference.

Fast-folding proteins use TICA. Fit the reference trajectory, then project the predicted ensemble and, optionally, a BioEmu ensemble:

```bash
python benchmarks/bioemu-benchmark/fit_tica.py /PATH/TO/REFERENCE \
    --output-dir /PATH/TO/TICA

python benchmarks/bioemu-benchmark/project_tica.py \
    --model-dir /PATH/TO/TICA \
    --sample-dir /PATH/TO/SAMPLES \
    --bioemu-dir /PATH/TO/BIOEMU \
    --output-dir /PATH/TO/TICA_RESULTS
```

The landscape reads `projected_data` from `{name}_ref.npz`, `{name}_pred.npz`, and `{name}_bioemu.npz`. MAE and coverage read `test_case`, `mae`, and `coverage` from `results_metrics.csv`. Full flags are in [README4bioemu.md](../bioemu-benchmark/README4bioemu.md), section 4.

MD-emulation cases use the BioEmu contact projection instead:

```bash
python benchmarks/bioemu-benchmark/analyze_md_emulation.py \
    --sample_dir /PATH/TO/SAMPLES \
    --output_dir /PATH/TO/MD_RESULTS
```

The landscape reads `results_projections.npz`. MAE and coverage read `results_metrics.csv`. Details are in [README4bioemu.md](../bioemu-benchmark/README4bioemu.md), section 2.

### Panel g

Free energy against the native-contact fraction, for local unfolding.

```bash
python benchmarks/bioemu-benchmark/compare_to_multi_conf.py \
    /PATH/TO/PREDICTIONS --benchmark localunfolding
```

A native contact is a Cα pair under 8 Å with sequence separation of at least 3 residues in the reference. `{name}_contacts.npy` under `/PATH/TO/PREDICTIONS/processing/` stores the fraction of those pairs recovered in each model.

### Panel h

Local RMSD against the reference structures, for domain motion.

```bash
python benchmarks/bioemu-benchmark/compare_to_multi_conf.py \
    /PATH/TO/PREDICTIONS --benchmark domainmotion
```

`metrics_rmsd.pkl` in `/PATH/TO/PREDICTIONS` holds `local_rmsd` and `global_rmsd`. This panel reads `local_rmsd`.

## Figure 3

### Panels a, b

PeptoneDB-SAXS (n=439) and PeptoneDB-CS (n=659). G-score against RMSE, prior and reweighted. LOWESS is plot-side.

Follow [README4peptone.md](../peptonebench/README4peptone.md), sections 3 and 6. The SAXS run uses `PeptoneDB-SAXS.csv`. The chemical-shift run uses `PeptoneDB-CS.csv`. Each `PeptoneBench` table needs `label`, `RMSE`, and `rew_RMSE`. G-score is `mean_gscore` (or `mean_gscore_adopt2`) from the matching PeptoneDB CSV.

### Panel c

Cases from PeptoneDB-SAXS. Per-residue G-score, the intensity curve, and the pair distance distribution.

G-score along the sequence is the PeptoneDB `gscores` column for that label. The intensity curve and P(r) come from the [SAXS case curves and P(r)](#saxs-case-curves-and-pr) commands above. The intensity panel reads `q`, `Iq_exp`, `sigma`, and `Iq_pred`.

### Panel d

Cases from PeptoneDB-CS. Secondary Cα chemical shifts.

The forward model is the UCBShift run in [README4peptone.md](../peptonebench/README4peptone.md), section 3, which writes `UCBshift-{label}.csv`. The panel reads per-residue secondary Cα: experimental shift and ensemble-mean predicted shift, each minus the random-coil value, plus the uncertainty. Fields are `residues`, `exp_cs`, `uncertainties`, and `gen_cs`.

### Panels e, f

PeptoneDB-Integrative. Prior and reweighted SAXS RMSE, PRE RMSE, CS RMSE, and RDC Q-factor, per system and as a mean per method.

Follow [README4peptone.md](../peptonebench/README4peptone.md), sections 5 and 7:

```bash
python benchmarks/peptonebench/analyze_saxs_integrative.py \
    -i /PATH/TO/INTEGRATIVE_OUTPUT \
    -e /PATH/TO/PeptoneDB-Integrative

python benchmarks/peptonebench/analyze_cs_integrative.py \
    -i /PATH/TO/INTEGRATIVE_OUTPUT \
    -e /PATH/TO/PeptoneDB-Integrative \
    --bmrb_path /PATH/TO/cs_stat_aa_filt.csv \
    --info_path /PATH/TO/PeptoneDB-Integrative/PeptoneDB-Integrative.csv

python benchmarks/peptonebench/analyze_pre_integrative.py \
    -i /PATH/TO/INTEGRATIVE_OUTPUT \
    -e /PATH/TO/PeptoneDB-Integrative \
    --pre_path /PATH/TO/PDB_OUTPUT

python benchmarks/peptonebench/analyze_rdc_integrative.py \
    -i /PATH/TO/INTEGRATIVE_OUTPUT \
    -e /PATH/TO/PeptoneDB-Integrative \
    --rdc_path /PATH/TO/PDB_OUTPUT \
    --info_path /PATH/TO/PeptoneDB-Integrative/PeptoneDB-Integrative.csv
```

Metrics for the other methods are taken from [the Peptone deposition](https://zenodo.org/records/17306061/files/Predictions.tar.gz?download=1). The coupled scatters read the per-method mean of the prior and reweighted errors.

## Figure 4

### Panel b

SAXS RMSE distributions for IDPFold2, AlphaFold 3, Protenix v1, Protenix v2, and AF-multimer.

Run the PeptoneBench SAXS pipeline in [README4peptone.md](../peptonebench/README4peptone.md), sections 3 and 6, once per method. Each table needs `label` and `RMSE`. IDPFold2 and AlphaFold 3 also use `rew_RMSE`. The paired t-test is plot-side.

### Panels c, d

Cases from the SAXS benchmark. Intensity curve, pair distance distribution, distance map, and native-contact fraction.

Intensity and P(r) use the [SAXS case curves and P(r)](#saxs-case-curves-and-pr) commands. The distance map is `quick_analysis.py` with the reference structure:

```bash
python scripts/quick_analysis.py /PATH/TO/ENSEMBLE --reference-dir /PATH/TO/REFS
```

The map panel reads `ref`, `pred`, and `chains` from `{name}_contact_map.npy`. The contact-fraction panel reads `ref`, `all_pred`, and `chains`. The fraction itself, an inter-chain Cα pair under 12 Å in the reference that is also under 12 Å in the model, and the free-energy curve, are computed in the plot.

### Panel e

DockQ on the PNAS set. The reported value is the maximum DockQ over the ensemble.

```bash
python benchmarks/idr-multimer-benchmark/get_dockq.py \
    --traj_dir /PATH/TO/TRAJ_DIR \
    --ref_dir /PATH/TO/REF_DIR \
    --output /PATH/TO/dockq_scores.csv
```

Run it once per method. The CSV index is the case name and the column is `dockq`. Join to the PNAS table on `pdb_id`. That table's `source` column is `sd01` (Rg), `sd02` (IDRBind), or `sd03` (Fuzzy). Chain mapping is in [README4idr.md](../idr-multimer-benchmark/README4idr.md).

### Panels f, g, h

Disorder ratio on the PNAS set, and, for the fold-upon-binding cases, secondary Cα chemical shifts plus helix and coil content.

Disorder ratio is `disorder_ratio` from `quick_analysis.py`, run separately on the reference structures, the IDPFold2 ensembles, and the AlphaFold 3 structures. Join the three values to the PNAS `source` column from panel e. ROC, PRC, and the linear fit are plot-side. The binary label is a reference disorder ratio above 0.5.

Helix and coil are `helix_content`, `coil_content`, `full_helix_content`, and `full_coil_content` from the same `quick_analysis.py` run. For a peptide predicted against a complex, pass `--reference-dir` so `helix_content` / `coil_content` are the bound chain and `full_*` are the peptide.

Secondary Cα uses the same fields as [Figure 3, panel d](#panel-d).
