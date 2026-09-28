import argparse
import os
import pickle
import multiprocessing as mp
import warnings
from functools import partial

import numpy as np
import biotite.sequence as seq
import biotite.structure.io as strucio
import biotite.structure as struc
from biotite.sequence.align import align_optimal, SubstitutionMatrix
import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

ALIGNMENT_MATRIX = SubstitutionMatrix.std_protein_matrix()

RESI_THREE_TO_1 = {
    "3HP": "X", "4HP": "X", "5HP": "Q", "ABA": "A", "ACE": "X", "AIB": "A",
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "ASX": "B", "AYA": "A",
    "BMT": "T", "BOC": "X", "CBX": "X", "CEA": "C", "CGU": "E", "CME": "C",
    "CRO": "TYG", "CSD": "C", "CSO": "C", "CSS": "C", "CSW": "C", "CSX": "C",
    "CXM": "M", "CYS": "C", "CYX": "C", "DAL": "A", "DAR": "R", "DCY": "C",
    "DGL": "E", "DGN": "Q", "DHI": "H", "DIL": "I", "DIV": "V", "DLE": "L",
    "DLY": "K", "DPN": "F", "DPR": "P", "DSG": "N", "DSN": "S", "DSP": "D",
    "DTH": "T", "DTR": "W", "DTY": "Y", "DVA": "V", "FME": "M", "FOR": "X",
    "GLN": "Q", "GLU": "E", "GLX": "Z", "GLY": "G", "HID": "H", "HIE": "H",
    "HIP": "H", "HIS": "H", "HYP": "P", "ILE": "I", "IVA": "X", "KCX": "K",
    "LEU": "L", "LLP": "K", "LYS": "K", "MET": "M", "MLE": "L", "MSE": "M",
    "MVA": "V", "NH2": "X", "NLE": "L", "NLW": "L", "OCS": "C", "ORN": "A",
    "PCA": "Q", "PHE": "F", "PRO": "P", "PSW": "U", "PTR": "Y", "PVL": "X",
    "PYL": "O", "SAR": "G", "SEC": "U", "SEP": "S", "SER": "S", "STY": "Y",
    "THR": "T", "TPO": "T", "TPQ": "Y", "TRP": "W", "TYR": "Y", "TYS": "Y",
    "UNK": "X", "VAL": "V",
}


def main():
    parser = argparse.ArgumentParser(
        description="Rg, end-to-end distance, simplified DSSP, and Cα distance maps."
    )
    parser.add_argument(
        "pred_dir",
        help="Directory of multi-model PDB files, one {name}.pdb per system.",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Optional directory of {name}.pdb reference structures. "
        "Adds the reference distance matrix. When the prediction is one chain "
        "and the reference has several, DSSP is mapped onto the best-matching chain.",
    )
    args = parser.parse_args()

    pred_dir = args.pred_dir
    assert os.path.isdir(pred_dir), f"Prediction directory {pred_dir} does not exist."
    if args.reference_dir is not None:
        assert os.path.isdir(args.reference_dir), (
            f"Reference directory {args.reference_dir} does not exist."
        )

    system_names = sorted(
        f.replace(".pdb", "") for f in os.listdir(pred_dir) if f.endswith(".pdb")
    )
    assert system_names, f"No PDB files in {pred_dir}."

    _process_fn = partial(
        process_fn, pred_dir=pred_dir, reference_dir=args.reference_dir
    )
    if os.cpu_count() > 1 and len(system_names) > 1:
        with mp.Pool(os.cpu_count()) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap_unordered(_process_fn, system_names),
                    total=len(system_names),
                )
            )
    else:
        results = []
        for system in tqdm.tqdm(system_names):
            results.append(_process_fn(system))

    results = [result for result in results if result is not None]
    results = consolidate_results(results)
    with open(os.path.join(pred_dir, "metrics.pkl"), "wb") as handle:
        pickle.dump(results, handle)


def process_fn(system, pred_dir, reference_dir=None):
    """Rg, end-to-end distance, DSSP fractions, and a Cα distance map for one system."""
    pred_path = os.path.join(pred_dir, f"{system}.pdb")
    predict = strucio.load_structure(pred_path)

    rg_predict = rg(predict)
    re2e_predict = re2e(predict)
    contact = ca_distance_map(predict)

    ref_path = None
    map_onto_complex = False
    if reference_dir is not None:
        candidate = os.path.join(reference_dir, f"{system}.pdb")
        if os.path.isfile(candidate):
            ref_path = candidate
            reference = strucio.load_structure(candidate)
            contact["ref"] = ca_distance_matrix(first_model(reference))
            map_onto_complex = len(contact["chains"]) == 1 and n_chains(reference) > 1

    np.save(os.path.join(pred_dir, f"{system}_contact_map.npy"), contact)

    helix_content, coil_content, disorder_ratio = dssp_fractions(pred_path)
    full_helix_content = helix_content
    full_coil_content = coil_content
    if map_onto_complex and helix_content is not None:
        mapped = map_peptide_dssp(pred_path, ref_path)
        if mapped is not None:
            helix_content, coil_content, full_helix_content, full_coil_content = mapped

    return {
        "name": system,
        "rg_predict": rg_predict,
        "re2e_predict": re2e_predict,
        "helix_content": helix_content,
        "coil_content": coil_content,
        "full_helix_content": full_helix_content,
        "full_coil_content": full_coil_content,
        "disorder_ratio": disorder_ratio,
    }


def rg(structures):
    return struc.gyration_radius(structures)


def re2e(structures):
    distances = []
    models = [structures] if structures.coord.ndim == 2 else structures
    for model in models:
        model = model[model.atom_name == "CA"]
        coords = model.coord
        coords_diff = coords[0, :] - coords[-1, :]
        distances.append(np.linalg.norm(coords_diff))
    return np.array(distances)


def first_model(structures):
    if structures.coord.ndim == 2:
        return structures
    return structures[0]


def n_chains(structures):
    model = first_model(structures)
    return len(set(np.asarray(model.chain_id).tolist()))


def chain_lengths(model):
    """Residue counts per chain, in PDB order, counted on Cα atoms."""
    ca = model[model.atom_name == "CA"]
    if len(ca) == 0:
        return []
    lengths = []
    start = 0
    chain_ids = ca.chain_id
    for index in range(1, len(chain_ids) + 1):
        if index == len(chain_ids) or chain_ids[index] != chain_ids[start]:
            lengths.append(int(index - start))
            start = index
    return lengths


def ca_distance_matrix(model):
    """Cα–Cα distances in Å for one model."""
    ca = model[model.atom_name == "CA"]
    coords = np.asarray(ca.coord, dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


def ca_distance_map(structures):
    models = [structures] if structures.coord.ndim == 2 else list(structures)
    all_pred = np.stack([ca_distance_matrix(model) for model in models], axis=0)
    return {
        "pred": all_pred.mean(axis=0).astype(np.float32),
        "all_pred": all_pred,
        "chains": np.asarray(chain_lengths(models[0]), dtype=np.int32),
    }


def dssp_codes(pdb_path):
    """Simplified DSSP codes, shape (n_models, n_residues). None if DSSP cannot run."""
    try:
        import mdtraj as md
    except ImportError as exc:
        print(f"DSSP skipped for {pdb_path}: {exc}")
        return None

    try:
        trajectory = md.load(pdb_path)
        return md.compute_dssp(trajectory, simplified=True)
    except Exception as exc:
        print(f"DSSP skipped for {pdb_path}: {exc}")
        return None


def _fraction(codes, symbol):
    if codes is None:
        return None
    return np.mean(codes == symbol, axis=0)


def dssp_fractions(pdb_path):
    codes = dssp_codes(pdb_path)
    if codes is None:
        return None, None, np.nan
    helix = _fraction(codes, "H")
    coil = _fraction(codes, "C")
    disorder = float(np.mean(np.mean(codes == "C", axis=1)))
    return helix, coil, disorder


def _sequence_from_residues(residues):
    letters = []
    for residue in residues:
        name = residue.name if hasattr(residue, "name") else residue
        letters.append(RESI_THREE_TO_1.get(str(name).upper(), "X"))
    return seq.ProteinSequence("".join(letters))


def map_peptide_dssp(peptide_path, complex_path):
    """Map peptide DSSP onto the best-scoring chain of a complex.

    Returns reference helix, reference coil, peptide helix, peptide coil on the
    aligned residues. ``helix_content`` is the bound reference and
    ``full_helix_content`` is the peptide prediction.
    """
    try:
        import mdtraj as md
    except ImportError:
        return None

    try:
        peptide = md.load(peptide_path)
        complex_traj = md.load(complex_path)
        peptide_dssp = md.compute_dssp(peptide, simplified=True)
        complex_dssp = md.compute_dssp(complex_traj, simplified=True)
    except Exception as exc:
        print(f"DSSP mapping skipped for {peptide_path}: {exc}")
        return None

    peptide_residues = list(peptide.topology.residues)
    peptide_sequence = _sequence_from_residues(peptide_residues)
    scores = {}
    local_indices = {}
    chain_residue_indices = {}
    for chain in complex_traj.topology.chains:
        residues = list(chain.residues)
        if not residues:
            continue
        chain_sequence = _sequence_from_residues(residues)
        alignment = align_optimal(
            peptide_sequence, chain_sequence, ALIGNMENT_MATRIX, local=False
        )[0]
        scores[chain.index] = alignment.score
        trace = alignment.trace
        matched = (trace[:, 0] != -1) & (trace[:, 1] != -1)
        local_indices[chain.index] = (trace[matched, 0], trace[matched, 1])
        chain_residue_indices[chain.index] = [residue.index for residue in residues]

    if not scores:
        return None

    best_chain = max(scores, key=scores.get)
    peptide_idx, chain_idx = local_indices[best_chain]
    if len(peptide_idx) == 0:
        return None
    ref_columns = np.asarray(chain_residue_indices[best_chain], dtype=int)[chain_idx]
    ref_codes = complex_dssp[:, ref_columns]
    pred_codes = peptide_dssp[:, peptide_idx]
    return (
        _fraction(ref_codes, "H"),
        _fraction(ref_codes, "C"),
        _fraction(pred_codes, "H"),
        _fraction(pred_codes, "C"),
    )


def consolidate_results(results):
    """Consolidate a list of results into a dictionary of lists."""
    keys = results[0].keys() if results else []
    return {key: [result[key] for result in results] for key in keys}


if __name__ == "__main__":
    main()
