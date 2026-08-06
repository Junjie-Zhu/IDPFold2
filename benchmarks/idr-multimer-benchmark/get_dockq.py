import os
import warnings
import multiprocessing as mp
import tempfile
from itertools import permutations

import biotite.structure as struc
import biotite.structure.info as strucinfo
import biotite.structure.io as strucio
import biotite.sequence as seq
from biotite.sequence.align import SubstitutionMatrix, align_optimal
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
TMP_ROOT = './tmp'
ALIGNMENT_MATRIX = SubstitutionMatrix.std_protein_matrix()
TRAJECTORY_NAMES = ('traj_no_clash.xtc', 'traj.xtc', 'traj.dcd')


def calculate_dockq(model, native):
    from DockQ.DockQ import run_on_all_native_interfaces

    results = run_on_all_native_interfaces(model, native)
    return results[1]


def _first_model(structure):
    if isinstance(structure, struc.AtomArrayStack):
        return structure[0]
    return structure


def _iter_models(structure):
    if isinstance(structure, struc.AtomArrayStack):
        for model in structure:
            yield model
    else:
        yield structure


def _one_letter_code(res_name):
    try:
        return strucinfo.one_letter_code(str(res_name))
    except (KeyError, TypeError):
        return 'X'


def _chain_sequences(model):
    model = _first_model(model)
    if model.array_length() == 0:
        return {}

    amino_mask = struc.filter_amino_acids(model)
    protein = model[amino_mask & (model.hetero == False)]
    sequences = {}
    residue_starts = struc.get_residue_starts(protein, add_exclusive_stop=True)

    for start, end in zip(residue_starts[:-1], residue_starts[1:]):
        residue = protein[start:end]
        chain_id = str(residue.chain_id[0])
        sequences.setdefault(chain_id, []).append(_one_letter_code(residue.res_name[0]))

    return {chain_id: ''.join(residues) for chain_id, residues in sequences.items()}


def _reference_ab_chain_map(reference_model, chain_map=None):
    reference_sequences = _chain_sequences(reference_model)

    if chain_map is not None:
        if 'A' not in chain_map or 'B' not in chain_map:
            return None
        if chain_map['A'] not in reference_sequences or chain_map['B'] not in reference_sequences:
            return None
        return {'A': chain_map['A'], 'B': chain_map['B']}

    chains = list(reference_sequences)
    if 'A' in reference_sequences and 'B' in reference_sequences:
        return {'A': 'A', 'B': 'B'}
    if len(chains) < 2:
        return None
    return {'A': chains[0], 'B': chains[1]}


def _alignment_stats(reference_sequence, prediction_sequence):
    if not reference_sequence or not prediction_sequence:
        return None

    alignment = align_optimal(
        seq.ProteinSequence(reference_sequence),
        seq.ProteinSequence(prediction_sequence),
        ALIGNMENT_MATRIX,
    )[0]
    trace = alignment.trace
    matched = trace[(trace[:, 0] != -1) & (trace[:, 1] != -1)]
    if matched.size == 0:
        return None

    ref_indices = matched[:, 0].astype(int)
    pred_indices = matched[:, 1].astype(int)
    identities = sum(
        reference_sequence[ref_i] == prediction_sequence[pred_i]
        for ref_i, pred_i in zip(ref_indices, pred_indices)
    )
    length_delta = abs(len(reference_sequence) - len(prediction_sequence))
    return identities, len(ref_indices), -length_delta


def assign_ab_chains_by_sequence(prediction_model, reference_model, chain_map=None):
    reference_map = _reference_ab_chain_map(reference_model, chain_map=chain_map)
    if reference_map is None:
        return None

    reference_sequences = _chain_sequences(reference_model)
    prediction_sequences = _chain_sequences(prediction_model)
    if len(prediction_sequences) < 2:
        return None

    labels = ('A', 'B')
    best_assignment = None
    best_score = None
    for prediction_ids in permutations(prediction_sequences, len(labels)):
        total_identities = 0
        total_matched = 0
        total_length_score = 0
        exact_label_matches = 0

        for label, prediction_id in zip(labels, prediction_ids):
            reference_id = reference_map[label]
            stats = _alignment_stats(reference_sequences[reference_id], prediction_sequences[prediction_id])
            if stats is None:
                break

            identities, matched, length_score = stats
            total_identities += identities
            total_matched += matched
            total_length_score += length_score
            exact_label_matches += int(prediction_id == label or prediction_id == reference_id)
        else:
            score = (total_identities, total_matched, total_length_score, exact_label_matches)
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = {label: prediction_id for label, prediction_id in zip(labels, prediction_ids)}

    return best_assignment


def _rename_to_ab(model, chain_map=None):
    if chain_map is not None:
        if 'A' not in chain_map or 'B' not in chain_map:
            return None
        keep_mask = np.isin(model.chain_id, [chain_map['A'], chain_map['B']])
        model = model[..., keep_mask]
        if model.array_length() == 0:
            return None

        orig_chain_ids = model.chain_id.copy()
        model.chain_id[orig_chain_ids == chain_map['A']] = 'A'
        model.chain_id[orig_chain_ids == chain_map['B']] = 'B'
        return model

    chains = struc.get_chains(model)
    if len(chains) < 2:
        return None
    orig_chain_ids = model.chain_id.copy()
    model.chain_id[orig_chain_ids == chains[0]] = 'A'
    model.chain_id[orig_chain_ids == chains[1]] = 'B'
    return model


def find_trajectory_file(traj_path):
    for traj_name in TRAJECTORY_NAMES:
        candidate = os.path.join(traj_path, traj_name)
        if os.path.exists(candidate):
            return candidate
    return None


def _split_traj_to_models(traj_path, tmp_dir, native_model, chain_map):
    if traj_path.lower().endswith('.pdb'):
        traj = strucio.load_structure(traj_path)
        models = list(_iter_models(traj))
        prediction_chain_map = assign_ab_chains_by_sequence(models[0], native_model, chain_map=chain_map)
        if prediction_chain_map is None:
            print(f'Skip {traj_path}: could not assign predicted chains by sequence')
            return 0

        for i, model in enumerate(models):
            model = _rename_to_ab(model, prediction_chain_map)
            if model is None:
                return 0
            strucio.save_structure(f'{tmp_dir}/test_model_{i}.pdb', model)
        return len(models)

    if os.path.isdir(traj_path):
        top_path = os.path.join(traj_path, 'topology.pdb')
        trajectory_path = find_trajectory_file(traj_path)
        if not os.path.exists(top_path):
            print(f'Skip {traj_path}: topology.pdb not found')
            return 0
        if trajectory_path is None:
            print(f'Skip {traj_path}: no supported trajectory file found')
            return 0

        import mdtraj as md

        topology = _first_model(strucio.load_structure(top_path))
        prediction_chain_map = assign_ab_chains_by_sequence(topology, native_model, chain_map=chain_map)
        if prediction_chain_map is None:
            print(f'Skip {traj_path}: could not assign predicted chains by sequence')
            return 0

        keep_mask = np.isin(topology.chain_id, [prediction_chain_map['A'], prediction_chain_map['B']])
        renamed_topology = _rename_to_ab(topology, prediction_chain_map)
        if renamed_topology is None:
            return 0
        traj = md.load(trajectory_path, top=top_path)

        if traj.n_atoms != topology.array_length():
            print(f'Skip {traj_path}: atom count mismatch between trajectory and topology')
            return 0

        for i in range(traj.n_frames):
            model = renamed_topology.copy()
            model.coord = traj.xyz[i, keep_mask] * 10.0  # mdtraj stores coordinates in nm
            strucio.save_structure(f'{tmp_dir}/test_model_{i}.pdb', model)
        return traj.n_frames

    print(f'Skip {traj_path}: unsupported trajectory format')
    return 0


def process_fn(traj, ref, chain_map):
    os.makedirs(TMP_ROOT, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'{os.path.basename(traj)}_', dir=TMP_ROOT) as tmp_dir:
        # reindex native structure, only keep the reference A/B chains
        try:
            native = strucio.load_structure(ref)
        except Exception as e:
            print(f'Error loading native structure from {ref}: {e}')
            return None

        # only keep the protein elements
        native = native[..., native.hetero == False]
        native_model = _first_model(native)
        reference_chain_map = _reference_ab_chain_map(native_model, chain_map=chain_map)
        if reference_chain_map is None:
            print(f'Skip {ref}: could not identify native A/B chains')
            return None

        # split trajectory into separate models after sequence-based chain assignment
        n_models = _split_traj_to_models(traj, tmp_dir, native_model, chain_map=reference_chain_map)
        if n_models == 0:
            return None

        native = _rename_to_ab(native, chain_map=reference_chain_map)
        if native is None or native.array_length() == 0:
            return None

        if isinstance(native, struc.AtomArray):
            native_models = [native]
        else:
            native_models = native

        dockq = []
        from DockQ.DockQ import load_PDB

        test_models = [load_PDB(f'{tmp_dir}/test_model_{i}.pdb') for i in range(n_models)]
        for j, nat_model in enumerate(native_models):
            native_file = f'{tmp_dir}/native_model_{j}.pdb'
            strucio.save_structure(native_file, nat_model)
            loaded_native = load_PDB(native_file)

            for loaded_model in test_models:
                dockq_score = calculate_dockq(loaded_model, loaded_native)
                dockq.append(dockq_score)

        if len(dockq) == 0:
            return None
        return float(np.max(dockq))


def _parse_chain_id_dict(path):
    if not os.path.exists(path):
        return {}

    chain_id_map = pd.read_csv(path)
    chain_id_dict = {}
    for _, row in chain_id_map.iterrows():
        case = str(row['case'])
        chain_keys = str(row['chain_keys']).split(':')
        chain_ids = str(row['chain_ids']).split(':')
        if len(chain_keys) != len(chain_ids):
            continue
        chain_id_dict[case] = {chain_keys[i]: chain_ids[i] for i in range(len(chain_keys))}
    return chain_id_dict


def _mp_fn(task):
    traj, traj_dir, ref_dir, chain_map = task
    traj_path = os.path.join(traj_dir, traj)
    ref_path = os.path.join(ref_dir, traj + '.pdb')
    if not os.path.exists(ref_path):
        return None
    dockq = process_fn(traj_path, ref_path, chain_map)
    if dockq is None:
        return None
    return traj, dockq


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_dir', type=str, default='./aa_multimers')
    parser.add_argument('--ref_dir', type=str, default='/lustre/home/acct-clschf/clschf/jjzhu/datasets/IDPFold_mul_dataset/test_set/multimers/pnas_multimers/pdb/')
    parser.add_argument('--chain_id_dict', type=str, default='./multimer_chain_id.csv')
    args = parser.parse_args()
    
    traj_dir = args.traj_dir
    ref_dir = args.ref_dir
    chain_id_dict = _parse_chain_id_dict(args.chain_id_dict)

    trajs = [i.replace('.pdb', '') for i in os.listdir(ref_dir) if i.endswith('.pdb')]
    tasks = []
    for traj in trajs:
        tasks.append((traj, traj_dir, ref_dir, chain_id_dict.get(traj)))

    # Pre-filter missing trajectory inputs without mutating while iterating.
    filtered_tasks = []
    for traj, task_traj_dir, task_ref_dir, chain_map in tasks:
        traj_base = os.path.join(task_traj_dir, traj)
        top_path = os.path.join(traj_base, 'topology.pdb')
        trajectory_path = find_trajectory_file(traj_base)
        if trajectory_path is None:
            print(f'Skip {traj}: no supported trajectory file found')
            continue
        if not os.path.exists(top_path):
            print(f'Skip {traj}: topology.pdb not found')
            continue
        filtered_tasks.append((traj, task_traj_dir, task_ref_dir, chain_map))
    tasks = filtered_tasks

    dqs = {}
    if os.cpu_count() == 1:
        for traj, task_traj_dir, task_ref_dir, chain_map in tqdm(tasks):
            traj_path = os.path.join(task_traj_dir, traj)
            ref_path = os.path.join(task_ref_dir, traj + '.pdb')
            if not os.path.exists(ref_path):
                continue
            dockq = process_fn(traj_path, ref_path, chain_map)
            if dockq is not None:
                dqs[traj] = dockq
    else:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(_mp_fn, tasks), total=len(tasks)))
            for result in results:
                if result is None:
                    continue
                traj, dockq = result
                dqs[traj] = dockq

    dqs = pd.DataFrame.from_dict(dqs, orient='index', columns=['dockq'])
    dqs.to_csv('./dockq_scores.csv')


if __name__ == '__main__':
    main()
