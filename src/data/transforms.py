# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import gzip
import os
import pickle
import re
from collections import defaultdict
from math import prod
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
import torch
import wget
from loguru import logger
from scipy.spatial.transform import Rotation as Scipy_Rotation
from torch_geometric import transforms as T
from torch_geometric.data import Data



def sample_uniform_rotation(shape=(), dtype=None, device=None) -> torch.Tensor:
    """Samples rotation matrices uniformly from SO(3).

    Args:
        shape: Batch dimensions for sampling multiple rotations
        dtype: Data type for the output tensor
        device: Device to place the output tensor on

    Returns:
        Tensor of shape [*shape, 3, 3] containing uniformly sampled rotation matrices
    """
    return torch.tensor(
        Scipy_Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


class CopyCoordinatesTransform(T.BaseTransform):
    """Creates a backup copy of coordinates before applying modifications.

    This transform copies the original coordinates to coords_unmodified before any
    other transformations (like noising or rotations) are applied.
    """

    def __call__(self, graph: Data) -> Data:
        """Copies coordinates to coords_unmodified.

        Args:
            graph: PyG Data object containing protein structure data

        Returns:
            Modified graph with coords_unmodified added
        """
        graph.coords_unmodified = graph.coords.clone()
        return graph


class ChainBreakPerResidueTransform(T.BaseTransform):
    """Identifies chain breaks in protein structures.

    Creates a binary mask indicating whether each residue has a chain break,
    determined by CA-CA distances exceeding a threshold.
    """

    def __init__(self, chain_break_cutoff: float = 4.0):
        """Initializes the transform.

        Args:
            chain_break_cutoff: Maximum allowed distance between consecutive CA atoms
                before considering it a chain break
        """
        self.chain_break_cutoff = chain_break_cutoff

    def __call__(self, graph: Data) -> Data:
        """Identifies chain breaks and adds mask to graph.

        Args:
            graph: PyG Data object containing protein structure

        Returns:
            Graph with added chain_breaks_per_residue mask
        """
        ca_coords = graph.coords[:, 1, :]
        ca_dists = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=1)
        chain_breaks_per_residue = ca_dists > self.chain_break_cutoff
        graph.chain_breaks_per_residue = torch.cat(
            (
                chain_breaks_per_residue,
                torch.tensor([False], dtype=torch.bool, device=chain_breaks_per_residue.device),
            )
        )
        return graph


class PaddingTransform(T.BaseTransform):
    """Pads tensors in graph to a specified maximum size.

    Ensures all tensors in the graph have consistent size by padding
    with a fill value up to max_size along the first dimension.
    """

    def __init__(self, max_size=256, fill_value=0):
        """Initializes the transform.

        Args:
            max_size: Target size for padding
            fill_value: Value to use for padding
        """
        self.max_size = max_size
        self.fill_value = fill_value

    def __call__(self, graph: Data) -> Data:
        """Applies padding to all applicable tensors in graph.

        Args:
            graph: PyG Data object to pad

        Returns:
            Graph with padded tensors
        """
        for key, value in graph:
            if isinstance(value, torch.Tensor):
                if value.dim() >= 1:
                    pad_dim = 0
                    graph[key] = self.pad_tensor(value, self.max_size, pad_dim, self.fill_value)
        return graph

    def pad_tensor(self, tensor, max_size, dim, fill_value=0):
        """Pads a single tensor to specified size.

        Args:
            tensor: Tensor to pad
            max_size: Target size
            dim: Dimension to pad
            fill_value: Value to use for padding

        Returns:
            Padded tensor
        """
        if tensor.size(dim) >= max_size:
            return tensor
        pad_size = max_size - tensor.size(dim)
        padding = [0] * (2 * tensor.dim())
        padding[2 * (tensor.dim() - 1 - dim) + 1] = pad_size
        return torch.nn.functional.pad(tensor, pad=tuple(padding), mode="constant", value=fill_value)

    def __repr__(self) -> str:
        """Get a string representation of the class.

        Returns:
            str: String representation of the class
        """
        return f"{self.__class__.__name__}(max_size={self.max_size}, fill_value={self.fill_value})"


class GlobalRotationTransform(T.BaseTransform):
    """Applies random global rotation to protein coordinates.

    Should be used as the first transform that modifies coordinates to maintain
    consistency in subsequent transformations.
    """

    def __init__(self, rotation_strategy: Literal["uniform"] = "uniform"):
        """Initializes the transform.

        Args:
            rotation_strategy: Method for sampling rotations. Currently only "uniform" supported
        """
        self.rotation_strategy = rotation_strategy

    def __call__(self, graph: Data) -> Data:
        """Applies random rotation to coordinates.

        Args:
            graph: PyG Data object containing protein structure

        Returns:
            Graph with rotated coordinates

        Raises:
            ValueError: If rotation_strategy is not supported
        """
        if self.rotation_strategy == "uniform":
            rot = sample_uniform_rotation(dtype=graph.coords.dtype, device=graph.coords.device)
        else:
            raise ValueError(f"Rotation strategy {self.rotation_strategy} not supported")
        graph.coords = torch.matmul(graph.coords, rot)
        return graph


class CATHLabelTransform(T.BaseTransform):
    """Adds CATH labels if available to the protein."""

    def __init__(self, root_dir: str):
        """Initialize the transform with the root directory of the CATH data.

        Args:
            root_dir (str): where the CATH data is/should be stored
        """
        self.root_dir = Path(root_dir)
        self.pdb_chain_cath_uniprot_url = (
            "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_cath_uniprot.tsv.gz"
        )
        self.cath_id_cath_code_url = (
            "http://download.cathdb.info/cath/releases/daily-release/newest/cath-b-newest-all.gz"
        )
        self.cath_id_cath_code_filename = Path(self.cath_id_cath_code_url).name
        self.pdb_chain_cath_uniprot_filename = Path(self.pdb_chain_cath_uniprot_url).name

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

        if not os.path.exists(self.root_dir / self.pdb_chain_cath_uniprot_filename):
            logger.info("Downloading Uniprot/PDB CATH map...")
            wget.download(self.pdb_chain_cath_uniprot_url, out=str(self.root_dir))

        if not os.path.exists(self.root_dir / self.cath_id_cath_code_filename):
            logger.info("Downloading CATH ID to CATH code map...")
            wget.download(self.cath_id_cath_code_url, out=str(self.root_dir))

        logger.info("Processing Uniprot/PDB CATH map...")
        self.pdbchain_to_cathid_mapping = self._parse_cath_id()
        logger.info("Processing CATH ID to CATH code map...")
        self.cathid_to_cathcode_mapping, self.cathid_to_segment_mapping = self._parse_cath_code()

    def __call__(self, graph: Data) -> Data:
        """Map each PDB chain to its CATH ID and CATH code.

        Args:
           graph (Data): A Data object containing a PDB chain ID.

        Returns:
            Data: A Data object with the CATH ID and CATH code mapped to each PDB chain.
        """
        cath_ids = self.pdbchain_to_cathid_mapping.get(graph.id, None)
        if cath_ids:
            cath_code = [self.cathid_to_cathcode_mapping.get(cath_id, None) for cath_id in cath_ids]
        else:
            cath_code = None
        if cath_code:  # check for list of Nones in cath code list
            graph.cath_code = cath_code
        else:
            graph.cath_code = []
        return graph

    def _parse_cath_id(self) -> Dict[str, str]:
        """Parse the CATH ID for all PDB chains.

        Args:
            None

        Returns:
            Dict[str, str]: Dictionary of PDB chain ID with their
            corresponding CATH ID.
        """
        pdbchain_to_cathid_mapping = defaultdict(list)
        with gzip.open(self.root_dir / self.pdb_chain_cath_uniprot_filename, "rt") as f:
            next(f)  # Skip header line
            for line in f:
                try:
                    pdb, chain, uniprot_id, cath_id = line.strip().split("\t")
                    key = f"{pdb}_{chain}"
                    pdbchain_to_cathid_mapping[key].append(cath_id)
                except ValueError as e:
                    logger.warning(e)
                    continue
        return pdbchain_to_cathid_mapping

    def _parse_cath_code(self) -> Dict[str, str]:
        """Parse CATH codes and segment information from the CATH database file.

        Processes the CATH database file to extract CATH IDs, codes, and segment information.
        Handles both single and multiple segment cases, parsing the chain and position
        information for each segment.

        Args:
            None

        Returns:
            tuple:
                - Dict[str, str]: Mapping of CATH IDs to their CATH codes
                - Dict[str, list]: Mapping of CATH IDs to lists of segment information tuples
                                Each tuple contains (chain, segment_start, segment_end)

        Raises:
            ValueError: If the line format is invalid or cannot be parsed
        """
        cathid_to_cathcode_mapping = {}
        cathid_to_segment_mapping = {}

        with gzip.open(self.root_dir / self.cath_id_cath_code_filename, "rt") as f:
            for line in f:
                try:
                    # Split line into components
                    cath_id, cath_version, cath_code, cath_segment_and_chain = line.strip().split()

                    # Process segments
                    cath_segments_and_chains = []
                    if "," in cath_segment_and_chain:
                        segments = cath_segment_and_chain.split(",")
                        for segment in segments:
                            cath_segments_and_chains.append(segment)
                    else:
                        cath_segments_and_chains.append(cath_segment_and_chain)

                    # Separate segments and chains
                    cath_segments = []
                    cath_chains = []
                    for item in cath_segments_and_chains:
                        segment, chain = item.split(":")
                        cath_segments.append(segment)
                        cath_chains.append(chain)

                    # Process start and end positions
                    cath_segments_start = []
                    cath_segments_end = []
                    for segment in cath_segments:
                        start, end = self.split_segment(segment)
                        cath_segments_start.append(start)
                        cath_segments_end.append(end)

                    # Store mappings
                    cathid_to_cathcode_mapping[cath_id] = cath_code

                    # Create segment info list
                    segment_info = []
                    for i in range(len(cath_chains)):
                        segment_info.append((cath_chains[i], cath_segments_start[i], cath_segments_end[i]))
                    cathid_to_segment_mapping[cath_id] = segment_info

                except ValueError as e:
                    logger.warning(e)
                    continue

        return cathid_to_cathcode_mapping, cathid_to_segment_mapping

    def split_segment(self, segment: str) -> Tuple[str, str]:
        """Split a segment into start position and end position.

        Handles cases where start or end position are negative numbers.

        Args:
            segment (str): segment description, for example `1-48` or `-2-36` or `1T-14M`.

        Returns:
            Tuple[str, str]: tuple containing start and end position, for example `(1, 48)` or `(-2, 36)` or `(1T, 14M)`.
        """
        # This regex pattern matches (potentially negative) numbers with potentially letters after them and separates segments by hyphen
        pattern = r"(-?\d+[A-Za-z]*)-(-?\d+[A-Za-z]*)"
        match = re.match(pattern, segment)
        if match:
            return match.groups()
        raise ValueError(f"Segment {segment} is not in the correct format")


class TEDLabelTransform(T.BaseTransform):
    """Adds CATH labels if available to the AFDB protein.

    Download ted_365m.domain_summary.cath.globularity.taxid.tsv.gz
    from https://zenodo.org/records/13908086
    and point to it via the file_path attribute.
    """

    def __init__(
        self,
        file_path,
        pkl_path,
        chunk_size=50000000,
    ):
        """Initialize the TEDLabelTransform.

        Args:
            file_path (str): Path to the TED domain summary file containing CATH labels.
                Set it to something like "<your-path>/ted_365m.domain_summary.cath.globularity.taxid.tsv".
            pkl_path (str): Base path for storing chunked pickle files of processed data.
                Set it to something like "<your-path>/afdb_to_cath_ted.pkl".
            chunk_size (int): Maximum number of samples to store in each pickle chunk.
                Defaults to 50000000.
        """
        self.file_path = file_path
        self.pkl_path = pkl_path
        self.chunk_size = chunk_size
        self.sample_to_cath = {}
        self._process_file()

    def _process_file(self):
        if self._pickle_exists():
            logger.info("AFDB CATH data already processed, loading now")
            self._load_pickles()
        else:
            logger.info("AFDB CATH data not processed yet, processing now")
            self._create_pickles()
            self._load_pickles()

    def _pickle_exists(self):
        return os.path.exists(f"{self.pkl_path}.0")

    def _create_pickles(self):
        """Process input file and create chunked pickle files mapping sample IDs to CATH codes.

        Reads the input file line by line, extracting sample IDs and their associated CATH codes.
        Creates multiple pickle files when the number of processed samples reaches chunk_size.
        Each pickle file contains a dictionary mapping sample IDs to lists of CATH codes.

        The function:
        - Processes the input file line by line
        - Extracts sample ID and CATH codes from each line
        - Groups CATH codes by sample ID in chunks
        - Saves each chunk to a separate pickle file when chunk_size is reached
        - Saves any remaining data in the final chunk

        Note:
        Uses self.file_path as input file path
        Uses self.chunk_size to determine when to create new pickle files
        Calls self._save_pickle() to save each chunk
        """
        sample_to_cath = {}
        counter = 0
        chunk_counter = 0
        with open(self.file_path) as file:
            for line in file:
                parts = line.strip().split("\t")
                full_sample_id = parts[0]
                cath_codes = parts[13].split(",") if parts[13] != "-" else []
                sample_id = "_".join(full_sample_id.split("_")[:-1])
                if cath_codes:
                    if sample_id not in sample_to_cath:
                        sample_to_cath[sample_id] = []
                        counter += 1
                        if counter % 1000000 == 0:
                            logger.info(f"Processed {counter} samples.")
                    sample_to_cath[sample_id].extend(cath_codes)
                if counter == self.chunk_size:
                    self._save_pickle(sample_to_cath, chunk_counter)
                    chunk_counter += 1
                    counter = 0
                    sample_to_cath = {}
        if sample_to_cath:
            self._save_pickle(sample_to_cath, chunk_counter)

    def _save_pickle(self, sample_to_cath, chunk_counter):
        """Save data to a pickle file with chunk number suffix.

        Args:
            sample_to_cath (dict): The data to be saved in pickle format.
            chunk_counter (int): Counter used to create unique file names for chunks.

        Returns:
            None

        Note:
            The function creates a pickle file with name pattern {self.pkl_path}.{chunk_counter}
            where chunk_counter is appended to the base pickle path.
        """
        pkl_path = f"{self.pkl_path}.{chunk_counter}"
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(sample_to_cath, pkl_file)

    def _load_pickles(self):
        """Load pickles from disk and merge them into one dictionary."""
        chunk_counter = 0
        while os.path.exists(f"{self.pkl_path}.{chunk_counter}"):
            with open(f"{self.pkl_path}.{chunk_counter}", "rb") as pkl_file:
                chunk_data = pickle.load(pkl_file)
                self.sample_to_cath.update(chunk_data)
            chunk_counter += 1

    def __call__(self, graph: Data) -> Data:
        """Call transform on sample.

        Args:
            graph (Data): protein graph

        Returns:
            Data: modified protein graph with CATH label
        """
        graph_id = graph.id
        _cath_code = self.sample_to_cath.get(graph_id, [])
        # For those only have CAT labels, pad them to CATH labels
        cath_code = []
        for code in _cath_code:
            if code.count(".") == 2:
                code = code + ".x"
            cath_code.append(code)
        graph.cath_code = cath_code
        return graph