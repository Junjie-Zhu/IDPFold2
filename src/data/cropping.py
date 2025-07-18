import numpy as np
import torch


def single_chain_truncate(atom_object, token_object, truncate_size=384):
    if token_object['token_index'].shape[0] <= truncate_size:
        return atom_object, token_object

    # Randomly select a chain ID
    chain_ids = np.unique(token_object['chain_index'])
    chain_id = np.random.choice(chain_ids)
    chain_indices = np.where(token_object['chain_index'] == chain_id)[0]

    # Crop if still too long
    if len(chain_indices) > truncate_size:
        crop_start = np.random.randint(0, len(chain_indices) - truncate_size + 1)
        crop_end = crop_start + truncate_size
        chain_indices = chain_indices[crop_start:crop_end]

    # Crop token_object for each key using the precomputed indices
    cropped_token_object = {k: v[chain_indices] for k, v in token_object.items()}

    # Determine atom cropping boundaries from token indices
    crop_atom_start = token_object['token_index'][chain_indices[0]]
    crop_atom_end = token_object['token_index'][chain_indices[-1]]
    crop_atom_mask = (atom_object['atom_to_token_index'] >= crop_atom_start) & \
                     (atom_object['atom_to_token_index'] <= crop_atom_end)

    # Crop atom_object for each key using the precomputed indices
    cropped_atom_object = {k: v[crop_atom_mask] for k, v in atom_object.items()}

    cropped_token_object['token_index'] -= np.min(cropped_token_object['token_index'])
    cropped_atom_object['atom_to_token_index'] -= np.min(cropped_atom_object['atom_to_token_index'])
    return cropped_atom_object, cropped_token_object


def contiguous_truncate(atom_object, token_object, truncate_size=384):
    if token_object['token_index'].shape[0] <= truncate_size:
        return atom_object, token_object

    # Prepare output dictionaries as lists for later concatenation
    cropped_atom_object = {k: [] for k in atom_object}
    cropped_token_object = {k: [] for k in token_object}

    # Precompute chain indices for each unique chain ID
    chain_ids = np.unique(token_object['chain_index'])
    chain_indices = {cid: np.where(token_object['chain_index'] == cid)[0] for cid in chain_ids}

    # Shuffle the chain IDs using numpy's in-place shuffle
    shuffled_chain_ids = chain_ids.copy()
    np.random.shuffle(shuffled_chain_ids)

    n_added = 0
    n_remaining = token_object['token_index'].shape[0]

    for cid in shuffled_chain_ids:
        indices = chain_indices[cid]
        chain_size = indices.shape[0]
        n_remaining -= chain_size

        # Determine crop size limits for the current chain
        crop_size_max = min(chain_size, truncate_size - n_added)
        crop_size_min = min(chain_size, max(0, truncate_size - n_added - n_remaining))
        crop_size = np.random.randint(crop_size_min, crop_size_max + 1)
        if crop_size <= 2:
            continue
        n_added += crop_size

        # Get crop indices for tokens
        crop_start = np.random.randint(0, chain_size - crop_size + 1)
        crop_end = crop_start + crop_size
        token_crop_indices = indices[crop_start:crop_end]

        # Crop token_object for each key using the precomputed indices
        for k, v in token_object.items():
            cropped_token_object[k].append(v[token_crop_indices])

        # Determine atom cropping boundaries from token indices
        crop_atom_start = token_object['token_index'][token_crop_indices[0]]
        crop_atom_end = token_object['token_index'][token_crop_indices[-1] + 1] \
            if (token_crop_indices[-1] + 1) < token_object['token_index'].shape[0] \
            else token_object['token_index'][token_crop_indices[-1]] + 1
        crop_atom_mask = (atom_object['atom_to_token_index'] >= crop_atom_start) & \
                         (atom_object['atom_to_token_index'] < crop_atom_end)
        for k, v in atom_object.items():
            cropped_atom_object[k].append(v[crop_atom_mask])

        # Stop if the desired total crop size is reached
        if n_added >= truncate_size:
            break

    # Concatenate the lists of arrays into single numpy arrays
    cropped_atom_object = {k: np.concatenate(v, axis=0) for k, v in cropped_atom_object.items()}
    cropped_token_object = {k: np.concatenate(v, axis=0) for k, v in cropped_token_object.items()}

    # Reindex token id and atom_to_token id
    token_id_mapping = {tid: idx for idx, tid in enumerate(cropped_token_object['token_index'])}
    cropped_token_object['token_index'] = np.array(
        [token_id_mapping[tid] for tid in cropped_token_object['token_index']])
    cropped_atom_object['atom_to_token_index'] = np.array(
        [token_id_mapping[tid] for tid in cropped_atom_object['atom_to_token_index']])

    return cropped_atom_object, cropped_token_object


def spatial_truncate(atom_object, token_object, truncate_size=384):
    if token_object['token_index'].shape[0] <= truncate_size:
        return atom_object, token_object

    # Prepare output dictionaries as lists for later concatenation
    cropped_atom_object = {k: [] for k in atom_object}
    cropped_token_object = {k: [] for k in token_object}

    _, unique_token_indices = np.unique(atom_object['atom_to_token_index'], return_index=True)
    atom_com = np.stack([atom_object['atom_com'][idx] for idx in unique_token_indices], axis=0)

    # Get a random center residue
    center_atom_index = np.random.randint(0, atom_com.shape[0])
    center_atom_coord = atom_com[center_atom_index]

    # Compute distances to the center residue
    distances = np.linalg.norm(atom_com - center_atom_coord, axis=1)
    truncate_dist = np.sort(distances)[truncate_size]
    crop_token_mask = distances <= truncate_dist

    # Crop token level features
    for k, v in token_object.items():
        cropped_token_object[k] = v[crop_token_mask]

    # Get atom indices for cropping
    cropped_token_index = cropped_token_object['token_index']
    cropped_atom_mask = np.isin(atom_object['atom_to_token_index'], cropped_token_index)

    # Crop atom level features
    for k, v in atom_object.items():
        cropped_atom_object[k] = v[cropped_atom_mask]

    # Reindex token id and atom_to_token id
    token_id_mapping = {tid: idx for idx, tid in enumerate(cropped_token_object['token_index'])}
    cropped_token_object['token_index'] = np.array(
        [token_id_mapping[tid] for tid in cropped_token_object['token_index']])
    cropped_atom_object['atom_to_token_index'] = np.array(
        [token_id_mapping[tid] for tid in cropped_atom_object['atom_to_token_index']])

    return cropped_atom_object, cropped_token_object


def single_chain_choice(data_object, truncate_size=384):
    atom_object = {
        'atom_positions': data_object['atom_positions'],
        'atom_to_token_index': data_object['atom_to_token_index'],
        'atom_com': data_object['atom_com'],
        'atom_mask': data_object['atom_mask'],

        'ref_positions': data_object['ref_positions'],
        'ref_element': data_object['ref_element'],
        'ref_atom_name_chars': data_object['ref_atom_name_chars'],
        'ref_com': data_object['ref_com'],
        'ref_space_uid': data_object['ref_space_uid'],
        'ref_structure': data_object['ref_structure'],
        'ref_mask': data_object['ref_mask'],
    }
    token_object = {
        'aatype': data_object['aatype'],
        'moltype': data_object['moltype'],
        'residue_index': data_object['residue_index'],
        'chain_index': data_object['chain_index'],
        'token_index': data_object['token_index'],
    }
    if token_object['token_index'].shape[0] <= truncate_size:
        return atom_object, token_object

    chain_ids = np.unique(token_object['chain_index'])
    chained_data = []
    for chain_id in chain_ids:
        chain_indices = torch.where(token_object['chain_index'] == chain_id)[0]

        # Crop token_object for each key using the precomputed indices
        cropped_token_object = {k: v[chain_indices] for k, v in token_object.items()}

        # Determine atom cropping boundaries from token indices
        crop_atom_start = token_object['token_index'][chain_indices[0]]
        crop_atom_end = token_object['token_index'][chain_indices[-1]]
        crop_atom_mask = (atom_object['atom_to_token_index'] >= crop_atom_start) & \
                         (atom_object['atom_to_token_index'] <= crop_atom_end)

        # Crop atom_object for each key using the precomputed indices
        cropped_atom_object = {k: v[crop_atom_mask] for k, v in atom_object.items()}

        cropped_token_object['token_index'] -= torch.min(cropped_token_object['token_index'])
        cropped_atom_object['atom_to_token_index'] -= torch.min(cropped_atom_object['atom_to_token_index'])

        cropped_atom_object.update(cropped_token_object)
        chained_data.append(cropped_atom_object)

    return chained_data

