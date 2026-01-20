# IDPFold2

![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

Implementation for [***Extending Conformational Ensemble Prediction to Multidomain Proteins and Protein Complex***](https://www.biorxiv.org/content/10.64898/2026.01.14.699584v1).

***Under construction***

## Description

***IDPFold2*** is a generative framework that models the heterogenous protein thermodynamics by integrating a Mixture-of-Experts architecture into the flow matching framework. The model is trained on a hybrid set of PDB, [mdCATH](https://github.com/compsciencelab/mdCATH), [IDRome-o](https://zenodo.org/records/17306061) and [AF-CALVADOS](https://github.com/KULL-Centre/_2025_buelow_AF-CALVADOS); and tested on [BioEmu-Benchmarks](https://github.com/microsoft/bioemu-benchmarks/tree/main/bioemu_benchmarks) and [PeptoneBench](https://github.com/PeptoneLtd/peptonebench/tree/main). 

This repository contains training and inference code, and useful scripts for evaluation.

## Table of Contents

* [Installation](#Installation)
* [Inference](#Inference)
  * [Monomers](#For-monomers)
  * [Multimers](#For-multimers)
* [Train](#Train)
  * [Preprocess PDB data](#Preprocess-PDB-data)
  * [Preprocess customized data](#Preprocess-customized-data)
  * [Train from scratch](#Train-from-scratch)
  * [Finetune](#Finetune)
* [Quick Evaluation](#Quick-Evaluation)
  * [Backmapping](#Backmapping)
  * [RMSD and Native Contact](#RMSD-and-Native-Contact)
  * [Reweighting](#Reweighting)
* [Contact](#Contact)
* [Acknowledgement](#Acknowledgement)

## Installation

**Fetch the project and install dependencies:**

```bash
git clone https://github.com/Junjie-Zhu/IDPFold2
cd IDPFold2

conda env create -f environment.yml
pip install fair-esm
pip install .
```

**Optional:** The torch implementation of MoE is used by default, you may also use the accelerated version from [MegaBlocks](https://github.com/databricks/megablocks/tree/main), which we provide a simplified version here.

```bash
# Optional: install megablocks
cd megablocks
pip install .
```

**Note:** 

* In some cases it will raise an undefined symbol error during installation, please refer to [this issue](https://github.com/databricks/megablocks/issues/159) for fixation. 
* The acceleration effect of MegaBlocks has not been tested on our model as we mainly performed inference on Ascend 910B, which did not support this package.  Nevertheless, using either torch or Megablocks version merely affect the predicted structure.

**Download weights from [Zenodo](https://zenodo.org/records/18239596).** 

* `IDPFold2_ema_0.999_260114.pth`: For **inference**, or EMA checkpoint for training.
* `IDPFold2_260114.pth`:  For training only.

## Inference

The model takes a `.csv` file as input, where monomers and multimers are handled seperately. The file should contain two columns `test_case` and `sequence`, denoting the name and sequence of target system respectively.

Directory to which the PLM embeddings are saved should be assigned. If no embedding file is found in the directory, the embeddings will be extracted and stored in the assigned directory.

### For monomers

```bash
python src/inference.py \
	prefix=MONOMER \
    ckpt_dir=/PATH/TO/CHECKPOINT/IDPFold2_ema_0.999_260114.pth \
    plm_emb_dir=./embeddings \
    csv_dir=/PATH/TO/INPUT/SEQUENCES \
    nsamples=100 \
    max_batch_length=6000 
```

**Important arguments:**

* `ckpt_dir`: Path to the fetched model weight, should be a `.pth` file.
* `plm_emb_dir`: Path to PLM embeddings. If not exist or empty, embeddings will be calculated and stored in the assigned directory.
* `csv_dir`: Path to the `.csv` file recording system names and sequences.
* `max_batch_length`: Maximum residue number to process on a single device. For example, `max_batch_length=6000` will force predicting 50 samples for a 120-residue protein each round. `6000` works for all test proteins in our implementation on Ascend 910B (64G).

Inference with multiple devices is supported with `torchrun`, but note that fixing the random seed in this case will produce exactly the same predictions from each device:

```bash
torchrun --nproc-per-node=4 src/inference.py
```

### For multimers

Inference for multimers mainly differs from that for monomer in the `.csv` file. The sequences should be connected by `:`. An example is provided in `data` directory.

```bash
python src/inference.py \
	prefix=MULTIMER \
    ckpt_dir=/PATH/TO/CHECKPOINT/IDPFold2_ema_0.999_260114.pth \
    plm_emb_dir=./embeddings \
    csv_dir=/PATH/TO/INPUT/SEQUENCES \
    nsamples=100 \
    max_batch_length=6000 \
    load_multimer=True
```

**Important arguments:**

* `load_multimer`: Currently monomers and multimers cannot be handled simultaneously. Add this tag for multimer inference.

## Train

For training you will have to preprocess the training dataset into `.pkl` files. We adapted this part mainly from [Proteina](https://github.com/NVIDIA-Digital-Bio/proteina).Two options are provided:

### Preprocess PDB data

```python
dataselector = PDBDataSelector(
    data_dir=args.data.data_dir,
    fraction=args.data.fraction,
    molecule_type=args.data.molecule_type,
    experiment_types=args.data.experiment_types,
    min_length=args.data.min_length,
    max_length=args.data.max_length,
    oligomeric_min=args.data.oligomeric_min,
    oligomeric_max=args.data.oligomeric_max,
    best_resolution=args.data.best_resolution,
    worst_resolution= args.data.worst_resolution,
    has_ligands=[],
    remove_ligands=[],
    remove_non_standard_residues=True,
    remove_pdb_unavailable=True,
    exclude_ids=[]
) if args.data.molecule_type is not None else None

# instantiate dataset
data_module = PDBDataModule(...)  # see Line 97 in src/train.py
data_module.prepare_data()
```

The code is annotated in `src/train.py` by default. Activating this part will allow automatic data download and preprocess for PDB.

**Important arguments:**

* `molecule_type: "protein"`: Type of molecule for which to select.
* `experiment_types: ["diffraction", "EM"]`: Other options are "NMR" and "other".
* `split_type: "sequence_similarity"`: Split sequences by sequence similarity clustering, other option is "random".

* `split_sequence_similarity: 0.5`: Clustering at 50% sequence similarity.
* `train_val_prop: [0.99, 0.01]`: Ratios to use for train and val splits

### Preprocess customized data

For simulation data or any data you want to use, make sure they are in `.pdb` or `.cif` format, determine certain sequence similarity and directly run `src/train.py`. See next part for detailed explanation of training arguments. 

The hybrid dataset was created manually in our implementation by concatenation the processed metadata files.

### Train from scratch

```bash
python src/train.py \
	task_prefix=HYBRID_TRAIN \
	batch_size=8 \
	epochs=500 \
	data.data_dir=/PATH/TO/DATASET \
	data.plm_emb_dir=/PATH/TO/EMBEDDING \
```

**Important arguments:**

* `data.data_dir`: The root dataset directory. This directory should contain all information required for model training as listed:
  * `raw/`: All structures used for model training.
  * `processed/`: Processed features, features will be created if not exists. 
  * `{data_dir}.csv`: Metadata for all training data, created after features are extracted.
  * `seq_{data_dir}.csv`: Sequences, used for further clustering.
  * `cluster_seqid_{split_sequence_similarity}_{data_dir}.tsv/fasta`: Cluster info.
* `data.plm_emb_dir`: Directory to PLM embeddings. For training you have to extract the embeddings first, you may refer to `scripts/get_esm_embedding.py` for this step.

You can find all arguments in `configs/train.yaml`. Distributed training is supported with `torchrun`, but **training across multiple machines is not supported**. Main reason for this is that device-level balance loss in MoE is not implemented, and training across machines may result in unexpected imbalanced expert assignment.

**Train with multimer data**

In our implementation, we first calculated inter-chain contacts for all downloaded PDB data and save as a `.csv` file. Multimers are assembled on the fly when assigning the following arguments:

```bash
python src/train.py \
	... \
	data.complex_dir=/PATH/TO/contacts.csv \
	data.complex_prop=0.8
```

### Finetune

If you want to finetune IDPFold2 from our pretrained checkpoints, both the model checkpoint and the EMA checkpoint are required (inference requires only the latter). 

```bash
python src/train.py \
	... \
	resume.ckpt_dir=/PATH/TO/CHECKPOINT/IDPFold2_260114.pth \
	resume.ema_dir=/PATH/TO/CHECKPOINT/IDPFold2_ema_0.999_260114.pth \
	resume.load_model_only=False
```

## Quick Evaluation

We provided post-processing scripts in the [scripts](scripts) directory, enabling quick evaluation of generated ensembles. We also provided some revised scripts from BioEmu-Benchmarks or PeptoneBench in the [benchmarks](benchmarks) directory, to calculate RMSD, native contacts, TiCA and reweighted SAXS/CS/PRE/RDC profiles. You may also refer to [Zenodo](https://zenodo.org/records/18239596) for plotting scripts (currently not uploaded, in preparation).

Radius of gyration (Rg) and end-to-end distance (Re2e) can be quickly calculated by the following command:

```bash
python scripts/quick_analysis.py /PATH/TO/GENERATED/ENSEMBLE
```

### Backmapping

To convert generated ensembles to all-atom structures, you have to install [cg2all](https://github.com/huhlim/cg2all) first.

```bash
convert_cg2all  # to test if cg2all is correctly installed

export OMP_NUM_THREAD=2
python scripts/_cg2all.py -i /PATH/TO/GENERATED/ENSEMBLE -o /PATH/TO/OUTPUT/STRUCTURES --num_proc 20
```

**Note: **You may have to adjust `OMP_NUM_THREAD` and `num_proc` (and `batch size`) for higher efficiency. Current setting works best in our practice with 40 cpu cores.

### RMSD and Native Contact

You have to first download information about the [BioEmu-Benchmarks](https://github.com/microsoft/bioemu-benchmarks/tree/main/bioemu_benchmarks) before calculating RMSDs and native contacts. Running the following script will extract both local and global RMSDs against reference structures, and fraction of native contacts for local unfolding cases.

```bash
python benchmarks/compare_to_multi_conf.py /PATH/TO/GENERATED/ENSEMBLE
```

### Reweighting

First download experimental data and useful information from [PeptoneBench (Zenodo link)](https://zenodo.org/record/17306061/files/PeptoneDBs.tar.gz), calculating SAXS/CS/PRE/RDC following the PeptoneBench protocols. Then you may use the following scripts for reweighting and analysis.

```bash
# first analyze SAXS and CS data
python analyze_saxs_integrative.py -i /PATH/TO/SAXS/PROFILES -e /PATH/TO/EXP/DATA
python analyze_cs_integrative.py \
    -i /PATH/TO/CS/PROFILES \
    -e /PATH/TO/EXP/DATA \
    --bmrb_path cs_stat_aa_filt.csv \
    --info_path PeptoneDB-Integrative.csv
    
# then analyze PRE and RDC data
python analyze_pre_integrative.py \
	-i /PATH/TO/SAXS/PROFILES \
	-e /PATH/TO/EXP/DATA \
	--pre_path /PATH/TO/PRE/PROFILES
python analyze_pre_integrative.py \
	-i /PATH/TO/CS/PROFILES \
	-e /PATH/TO/EXP/DATA \
	--rdc_path /PATH/TO/RDC/PROFILES
    --info_path PeptoneDB-Integrative.csv
```

**Important arguments:**

* `bmrb_path` in analyzing CS data: download from [BMRB Chemical Shift Statistics](https://bmrb.io/ref_info/)
* `i` in analyzing PRE/RDC should be path to SAXS/CS data respectively in order to use the pre-calculated reweighting information.
* `e` and `info_path` denotes information provided by PeptoneDB-Integrative.

## Contact

Please contact through email `shiroyuki@sjtu.edu.cn` or create an issue if you have any question.

## Acknowledgement

Thank [Z. Zheng](https://github.com/Immortals-33) and [Z. Fan](https://github.com/Zirui-Fan) for providing helpful discussions, and [J. Yu](https://github.com/yjyjyjy2016) for viewpoints on multidomain proteins.

The codebase is mainly constructed on [Proteina](https://github.com/NVIDIA-Digital-Bio/proteina).

The generated structures are backmapped with [cg2all](https://github.com/huhlim/cg2all).

Thank the open-source benchmarks [BioEmu-Benchmarks](https://github.com/microsoft/bioemu-benchmarks/tree/main/bioemu_benchmarks) and [PeptoneBench](https://github.com/PeptoneLtd/peptonebench/tree/main).

