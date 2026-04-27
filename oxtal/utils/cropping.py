import collections
import copy
import random
from collections import defaultdict
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure import get_chain_count, get_chains
from scipy.spatial.distance import cdist
import random
from collections import defaultdict, Counter

from protenix.data.constants import PRO_STD_RESIDUES
from protenix.data.tokenizer import TokenArray

_PRO_STD_RESIDUES_SET = set(PRO_STD_RESIDUES.values())
_BACKBONE_ATOMS = ["N", "CA", "C", "O"]
_N_BACKBONE_ATOMS = 4
_CA_INDEX = 1


class CropData:
    """
    Crop the data based on the given crop size and reference chain indices (asym_id).
    """

    def __init__(
        self,
        crop_size: int,
        ref_chain_indices: list[int],
        token_array: TokenArray,
        atom_array: AtomArray,
        method_weights: list[float] = [0.2, 0.4, 0.4],
        contiguous_crop_complete_lig: bool = False,
        spatial_crop_complete_lig: bool = False,
        drop_last: bool = False,
        remove_metal: bool = False,
    ) -> None:
        """
        Args:
            crop_size (int): The size of the crop to be sampled.
            ref_chain_indices (list[int]): The "asym_id_int" of the reference chains.
            token_array (TokenArray): The token array.
            atom_array (AtomArray): The atom array.
            method_weights (list[float]): The weights corresponding to these three cropping methods:
                                          ["ContiguousCropping", "SpatialCropping", "SpatialInterfaceCropping"].
            contiguous_crop_complete_lig: Whether to crop the complete ligand in ContiguousCropping method.

        """
        self.crop_size = crop_size
        self.ref_chain_indices = ref_chain_indices
        self.token_array = token_array
        self.atom_array = atom_array
        self.method_weights = method_weights
        self.cand_crop_methods = [
            "ContiguousCropping",
            "SpatialCropping",
            "SpatialInterfaceCropping",
        ]
        self.contiguous_crop_complete_lig = contiguous_crop_complete_lig
        self.spatial_crop_complete_lig = spatial_crop_complete_lig
        self.drop_last = drop_last
        self.remove_metal = remove_metal

    @staticmethod
    def select_by_token_indices(
        token_array: TokenArray,
        atom_array: AtomArray,
        selected_token_indices: torch.Tensor,
        msa_features: dict[str, np.ndarray] = None,
        template_features: dict[str, np.ndarray] = None,
    ) -> tuple[TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crop the token array, atom array, msa features and template features based on the selected token indices.

        Args:
            token_array (TokenArray): the input token array
            atom_array (AtomArray): the input atom array
            selected_token_indices (torch.Tensor): The indices of the tokens to be cropped.
            msa_feature (dict[str, np.ndarray]): The MSA features.
            template_feature (dict[str, np.ndarray]): The Template features.

        Returns:
            cropped_token_array (TokenArray): The cropped token array.
            cropped_atom_array (AtomArray): The cropped atom array.
            cropped_msa_features (dict[str, np.ndarray]): The cropped msa features.
            cropped_template_features (dict[str, np.ndarray]): The cropped template features.
        """
        cropped_token_array = copy.deepcopy(token_array[selected_token_indices])

        cropped_atom_indices = []
        totol_atom_num = 0
        for idx, token in enumerate(cropped_token_array):
            cropped_atom_indices.extend(token.atom_indices)
            centre_idx_in_token_atoms = token.atom_indices.index(token.centre_atom_index)
            token_atom_num = len(token.atom_indices)
            token.atom_indices = list(range(totol_atom_num, totol_atom_num + token_atom_num))
            token.centre_atom_index = token.atom_indices[centre_idx_in_token_atoms]
            totol_atom_num += token_atom_num

        cropped_atom_array = copy.deepcopy(atom_array[cropped_atom_indices])
        assert len(cropped_token_array) == selected_token_indices.shape[0]

        _selected_token_indices = selected_token_indices.tolist()
        # # crop msa
        cropped_msa_features, cropped_template_features = {}, {}
        return (
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
        )

    @staticmethod
    def remove_non_backbone_atoms(
        token_array: TokenArray,
        atom_array: AtomArray,
    ) -> tuple[TokenArray, AtomArray]:
        """
        Removes non-backbone atoms
        """
        # atoms_to_keep_mask will only be False for atoms which are for a protein residue
        # but which are not backbone atoms.
        atom_idx_offset = np.zeros(len(atom_array), dtype=int)
        atoms_to_keep_mask = np.ones(len(atom_array), dtype=bool)

        # Step 1: loop once to build up the atom_idx_offset
        for token in token_array:
            if token.value not in _PRO_STD_RESIDUES_SET:
                continue

            assert token.atom_names[:_N_BACKBONE_ATOMS] == _BACKBONE_ATOMS, (
                "The token should have its first four atoms equal to "
                + f"{_BACKBONE_ATOMS}, but the token atom names were ${token.atom_names}"
            )

            old_num_atoms = len(token.atom_indices)
            atom_idx_offset[token.atom_indices[-1] + 1 :] -= old_num_atoms - _N_BACKBONE_ATOMS
            atoms_to_keep_mask[token.atom_indices[_N_BACKBONE_ATOMS:]] = False
            token.atom_indices = token.atom_indices[:_N_BACKBONE_ATOMS]
            token.atom_names = token.atom_names[:_N_BACKBONE_ATOMS]

        # Filter out non-backbone protein atoms
        atom_array = atom_array[atoms_to_keep_mask]

        res_perms = atom_array.res_perm
        # Step 2: after atom_idx_offset is build, loop through the token_array
        #         again and apply the offset to the tokens.
        for token in token_array:
            atom_indices = token.atom_indices
            if not isinstance(atom_indices, np.ndarray):
                atom_indices = np.array(atom_indices)

            curr_centre_idx_idx = np.flatnonzero(atom_indices == token.centre_atom_index)
            assert len(curr_centre_idx_idx) == 1

            atom_indices += atom_idx_offset[atom_indices]
            token.centre_atom_index = atom_indices[curr_centre_idx_idx[0]]
            token.atom_indices = atom_indices.tolist()

            # Set the distogram rep atom to be the c-alpha for all the prot residues
            if token.value in _PRO_STD_RESIDUES_SET:
                atom_array.distogram_rep_atom_mask[token.atom_indices[_CA_INDEX]] = True
                for i, atom_idx in enumerate(token.atom_indices):
                    res_perm = res_perms[atom_idx]
                    idxs = (
                        [int(res_perm)]
                        if "_" not in res_perm
                        else [int(j) for j in res_perm.split("_")]
                    )

                    if max(idxs) > 3:
                        res_perms[atom_idx] = "_".join([str(i) for _ in range(len(idxs))])

        atom_array.set_annotation("res_perm", res_perms)
        return token_array, atom_array
