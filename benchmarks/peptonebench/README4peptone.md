# Running PeptoneBench for SAXS, CS, RDC, and PRE

This guide describes the workflow we used to calculate and analyze experimental errors for generated peptide ensembles against PeptoneBench observations.

The workflow has two stages that use different environments:

- `peptonebench`: the upstream PeptoneBench environment for SAXS and chemical-shift forward-model calculation.
- `peptone`: the local environment in this directory for RDC/PRE calculation and integrative analysis.

Use absolute paths in the commands below, or replace every `/PATH/TO/...` placeholder consistently before running.

## Directory and Input Conventions

The example commands use these placeholders:

- `/PATH/TO/ENSEMBLE`: directory containing generated ensembles.
- `/PATH/TO/CS_OUTPUT`: output directory for UCBShift chemical-shift predictions.
- `/PATH/TO/SAXS_OUTPUT`: output directory for Pepsi-SAXS predictions.
- `/PATH/TO/INTEGRATIVE_OUTPUT`: output directory for PeptoneDB-Integrative predictions.
- `/PATH/TO/PDB_OUTPUT`: protonated PDB directory produced by `addhydrogens.py`.
- `/PATH/TO/peptonebench`: clone of the upstream PeptoneBench repository.
- `/PATH/TO/IDPFold2`: clone of this repository.

Each ensemble should be stored in the structure expected by PeptoneBench:

```text
10036_1_1_1/
|-- topology.pdb
`-- traj_no_clash.xtc
```

## 1. Install External Software for Forward Models

Download the UCBShift weights from [Dryad](https://datadryad.org/stash/share/6vbrswTtNRcHk2vV3e6P1QGH1yYMhvdHDlauysTCObE). The downloaded file should be named `models.tgz`.

Install the external tools used by PeptoneBench:

```bash
# SPARTA+
wget https://spin.niddk.nih.gov/bax/software/SPARTA+/sparta+.tar.Z
tar xf sparta+.tar.Z

# Pepsi-SAXS
wget https://files.inria.fr/NanoDFiles/Website/Software/Pepsi-SAXS/Linux/3.0/Pepsi-SAXS-Linux.zip
unzip Pepsi-SAXS-Linux.zip

# Reduce
git clone https://github.com/rlabduke/reduce.git
cd reduce
make
cd ..

# CSpred
git clone https://github.com/THGLab/CSpred.git
mkdir -p CSpred/models
mv models.tgz CSpred/models/
tar -zxf CSpred/models/models.tgz -C CSpred/models/

# PeptoneBench
git clone https://github.com/PeptoneLtd/peptonebench.git
```

Download the PeptoneBench datasets:

```bash
cd /PATH/TO/peptonebench/datasets
source downloadDBs.sh
```

## 2. Create the Upstream PeptoneBench Environment

Use this environment for SAXS and CS forward-model calculation.

```bash
cd /PATH/TO/peptonebench/forward-models
conda env create -f env.yaml
conda activate peptonebench
```

Set paths to your local tool installations:

```bash
export PYTHONPATH=/PATH/TO/CSpred
export SPARTAP_DIR=/PATH/TO/SPARTA+
export PATH=/PATH/TO/reduce/bin:/PATH/TO/SPARTA+/bin:/PATH/TO/pepsi:$PATH
```

If DSSP or `mkdssp` fails during chemical-shift calculation, you may need to point Biopython/CSpred to the `mkdssp` executable available in your CSpred setup.

## 3. Run SAXS and CS Forward Models

Run these commands from `/PATH/TO/peptonebench/forward-models` with the `peptonebench` environment active.

Calculate chemical shifts with UCBShift:

```bash
python run_forward_model.py \
    -p UCBshift \
    -f /PATH/TO/ENSEMBLE \
    --output-dir /PATH/TO/CS_OUTPUT \
    --no-prepare-ensembles \
    --dataset /PATH/TO/peptonebench/datasets/PeptoneDB-CS/PeptoneDB-CS.csv
```

Calculate SAXS curves with Pepsi-SAXS:

```bash
python run_forward_model.py \
    -p Pepsi \
    -f /PATH/TO/ENSEMBLE \
    --output-dir /PATH/TO/SAXS_OUTPUT \
    --no-prepare-ensembles \
    --dataset /PATH/TO/peptonebench/datasets/PeptoneDB-SAXS/PeptoneDB-SAXS.csv \
    --saxs-data /PATH/TO/peptonebench/datasets/PeptoneDB-SAXS/sasbdb-clean_data \
    --pepsi-path /PATH/TO/pepsi/Pepsi-SAXS
```

To run on another PeptoneBench dataset, replace the `--dataset` argument. For example, use `PeptoneDB-Integrative/PeptoneDB-Integrative.csv` for the integrative dataset.

## 4. Create the Local RDC/PRE Environment

The following steps should be run in this repository, not in upstream `peptonebench/integrativeanalysis`.

```bash
cd /PATH/TO/IDPFold2/benchmarks/peptonebench
conda env create -f env.yaml
conda activate peptone
```

RDC calculation also requires `PALES`, which can be downloaded from [Markus Zweckstetter PALES](https://www3.mpibpc.mpg.de/groups/zweckstetter/_links/software_pales.htm). PALES is old software and may be difficult to run on 64-bit systems.

## 5. Calculate PRE and RDC Inputs

Run these commands from `/PATH/TO/IDPFold2/benchmarks/peptonebench` with the `peptone` environment active.

First add hydrogens to the generated ensembles:

```bash
python addhydrogens.py \
    -i /PATH/TO/ENSEMBLE \
    -o /PATH/TO/PDB_OUTPUT \
    -e /PATH/TO/peptonebench/datasets/PeptoneDB-Integrative/
```

Then calculate PRE and RDC values. Both scripts use the protonated PDB files from `addhydrogens.py`.

```bash
python calc_PRE.py \
    -i /PATH/TO/PDB_OUTPUT \
    -e /PATH/TO/peptonebench/datasets/PeptoneDB-Integrative/

python calc_RDC.py \
    -d /PATH/TO/PDB_OUTPUT \
    -p /PATH/TO/PALES
```

## 6. Analyze SAXS and CS Outputs

`PeptoneBench` assumes trajectories under the OUTPUT directory, so before analysis we have to copy our trajectory files accordingly:

```python
import os
import shutil
from tqdm import tqdm

input_dir = '/PATH/TO/ENSEMBLE'
output_dir = '/PATH/TO/SAXS_OUTPUT'
target_systems = [i.replace('.csv', '').replace('Pepsi-', '') for i in os.listdir(output_dir) 
                  if i.startswith('Pepsi-') and i.endswith('.csv')]
for sub_dirs in tqdm(target_systems):
    shutil.copy(f'{input_dir}/{sub_dirs}/topology.pdb', f'{output_dir}/{sub_dirs}.pdb')
    shutil.copy(f'{input_dir}/{sub_dirs}/traj_no_clash.xtc', f'{output_dir}/{sub_dirs}.xtc')
    
output_dir = '/PATH/TO/CS_OUTPUT'
target_systems = [i for i in os.listdir(input_dir) if not i.startswith('SAS')]
for sub_dirs in tqdm(target_systems):
    try:
        shutil.copy(f'{input_dir}/{sub_dirs}/topology.pdb', f'{output_dir}/{sub_dirs}.pdb')
        shutil.copy(f'{input_dir}/{sub_dirs}/traj_no_clash.xtc', f'{output_dir}/{sub_dirs}.xtc')
    except Exception as e:
        print(e)
        continue
```

For standard SAXS and CS benchmark outputs, use the PeptoneBench command-line analyzer:

```bash
PeptoneBench -f /PATH/TO/SAXS_OUTPUT
PeptoneBench -f /PATH/TO/CS_OUTPUT
PeptoneBench -f /PATH/TO/INTEGRATIVE_OUTPUT
```

`PeptoneBench` should automatically detect available SAXS, CS, or integrative dataset entries and compute the corresponding experimental errors.

## 7. Analyze Integrative SAXS, CS, PRE, and RDC

Run these commands from `/PATH/TO/IDPFold2/benchmarks/peptonebench` with the `peptone` environment active.

The CS analysis requires `cs_stat_aa_filt.csv`, available from [BMRB Chemical Shift Statistics](https://bmrb.io/ref_info/). Choose the filtered CSV for the 20 common amino acids.

```bash
python analyze_saxs_integrative.py \
    -i /PATH/TO/INTEGRATIVE_OUTPUT \
    -e /PATH/TO/peptonebench/datasets/PeptoneDB-Integrative

python analyze_cs_integrative.py \
    -i /PATH/TO/INTEGRATIVE_OUTPUT \
    -e /PATH/TO/peptonebench/datasets/PeptoneDB-Integrative \
    --bmrb_path /PATH/TO/cs_stat_aa_filt.csv \
    --info_path /PATH/TO/peptonebench/datasets/PeptoneDB-Integrative/PeptoneDB-Integrative.csv

python analyze_pre_integrative.py \
    -i /PATH/TO/INTEGRATIVE_OUTPUT \
    -e /PATH/TO/peptonebench/datasets/PeptoneDB-Integrative \
    --pre_path /PATH/TO/PDB_OUTPUT

python analyze_rdc_integrative.py \
    -i /PATH/TO/INTEGRATIVE_OUTPUT \
    -e /PATH/TO/peptonebench/datasets/PeptoneDB-Integrative \
    --rdc_path /PATH/TO/PDB_OUTPUT \
    --info_path /PATH/TO/peptonebench/datasets/PeptoneDB-Integrative/PeptoneDB-Integrative.csv
```

`/PATH/TO/PDB_OUTPUT` should be the same directory produced by `addhydrogens.py` and used for `calc_PRE.py` and `calc_RDC.py`.
