import random
from dataclasses import dataclass

import biotite.structure
import numpy as np
from scipy.spatial.distance import cdist

from oxtal.data.ccd import (
    _get_mask_ref,
    get_ccd_ref_info,
)
from oxtal.data.constants import (
    DNA_STD_RESIDUES,
    MASK_RESNAME,
    PRO_STD_RESIDUES,
    RNA_STD_RESIDUES,
    mmcif_restype_1to3,
)
from oxtal.data.parser import AddAtomArrayAnnot
from oxtal.data.tokenizer import AtomArrayTokenizer, TokenArray
from oxtal.utils.geometry import random_transform
from oxtal.utils.logger import get_logger

logger = get_logger(__name__)

_N_BACKBONE_ATOMS = 4


@dataclass
class MaskingResult:
    atom_array: biotite.structure.AtomArray
    token_array: TokenArray
    masked_res_indices: np.ndarray


class TaskManager:
    """
    TaskManager class is responsible for managing tasks. - Data transformer (post loading)

    * folding (input: sequence, mask structure)
    * inverse_folding (input: masked sequence, structure)
    * cogen (input: masked sequence, masked structure)

    * masking type (also percentage of function):
        * diffuse random masking (UNK for sequence, Gaussian noise for structure)
        * motif masking
            * proximity
            * chain-based
        * generation based (mask)

    Input:
    sequence, structure, task, mask_type, percentage_mask, schedule_masking

    output: sequence mask index

    """

    def __init__(
        self,
        tasks=None,
        percentage_mask="random",
        schedule_mask="random",
        mask_type="uniform",
        inference_task="unconditional_structure",
        inference_time=1.0,
        mask_na=False,
        transform_masked_ref_pos=True,
        ref_pos_augment=True,
    ):
        self.tasks = tasks  # Dictionary
        self.task_count = len(tasks)
        print(self.tasks)
        self.task_d = {t: self.tasks[t]["frac"] for t in self.tasks.keys()}
        self.percentage_mask = percentage_mask
        self.schedule_masking = schedule_mask
        self.mask_type = mask_type
        self.inference_task = inference_task
        self.inference_time = inference_time
        assert self.mask_type in [
            "uniform",
            "chain",
            "ligand_proximity",
        ], f"Invalid mask type {self.mask_type}"
        self.structure = ["label_dict", "coordinate"]  # n x 3 (all atoms), n is the same for
        self.type_of_seq = ["input_feature_dict", ["is_rna", "is_dna", "is_prot", "is_lig"]]
        self.backbone_atoms = ["CA", "C", "N", "O"]
        self.na_backbone_atoms = ["P", "C4'", "C1'"]
        self.mask_na = mask_na

        self._transform_masked_ref_pos = transform_masked_ref_pos
        self._ref_pos_augment = ref_pos_augment

    def _choose_task(self):
        return np.random.choice(list(self.task_d.keys()), p=list(self.task_d.values()))

    def mask(self, atom_array, task, t, distribution=None):
        """mask atom array, then return it"""
        self.preprocess_atom_array(atom_array)

        # If we're doing unconditional_structure as a task we want to mask every
        # residue in the sequence but do structure diffusion as normal. This
        # effectively means we want t=1 for sequence and t set normally for
        # structure. As such, if we're doing unconditional_structure, we set the
        # sequence t to 1 here with mask_t and leave the structure t unchanged.
        mask_t = t if task != "unconditional_structure" else 1.0
        if self.mask_type == "chain":
            masked_res_indices = self.chain_mask(atom_array, mask_t)
        elif self.mask_type == "ligand_proximity":
            masked_res_indices = self.ligand_proximity_mask(atom_array, mask_t)
        else:
            masked_res_indices = self.uniform_mask(mask_t)

        logger.debug(
            f"length of masked_res_indices: {len(masked_res_indices)} total: {len(self.sym_uid_hashes)}"
        )

        atom_array = self.mask_ref_information(atom_array, masked_res_indices)
        atom_array = self.mask_side_chains(atom_array, masked_res_indices)

        return MaskingResult(
            atom_array=atom_array,
            token_array=AtomArrayTokenizer(atom_array=atom_array).get_token_array(),
            masked_res_indices=masked_res_indices,
        )

    def uniform_mask(self, t):
        len_seq = len(self.sym_uuid_hashes)
        masked_sym_uuid_hash_indices = np.flatnonzero(np.random.rand(len_seq) < t)

        # perform symmetrization of masking
        masked_sym_uuid_hashes = self.sym_uuid_hashes[masked_sym_uuid_hash_indices]
        masked_sym_uuid_indices = np.where(np.isin(self.sym_uid_hashes, masked_sym_uuid_hashes))[0]
        # return np.intersect1d(masked_sym_uuid_indices, self.is_protein_indices)
        return np.intersect1d(masked_sym_uuid_indices, self.is_molecule_indices)

    def chain_mask(self, atom_array, t):
        """mask an entire chain, may mask more than the chain..."""
        unique_chains = np.unique(atom_array.chain_id)
        # choose a chain
        chain_chosen = np.random.choice(unique_chains)
        chain_indices = np.where(atom_array.chain_id == chain_chosen)[0]
        masked_sym_uid_hashes = self.sym_uid_hashes[chain_indices]
        masked_sym_uuid_hashes = np.unique(masked_sym_uid_hashes)
        masked_sym_uuid_indices = np.where(np.isin(self.sym_uid_hashes, masked_sym_uuid_hashes))[0]
        # return np.intersect1d(masked_sym_uuid_indices, self.is_protein_indices)
        return np.intersect1d(masked_sym_uuid_indices, self.is_molecule_indices)

    def ligand_proximity_mask(self, atom_array, t):
        """choose a ligand and mask residues that are close to it"""
        all_residues = np.unique(atom_array.res_name)
        non_protein_residues = np.setdiff1d(all_residues, np.array(list(PRO_STD_RESIDUES.keys())))
        # choose a residue
        if len(non_protein_residues) == 0:
            # logger.info("No non-protein residues found, uniform masking")
            # return self.uniform_mask(bioassembly_dict, t)
            # choose random coord as ligand
            avg_ligand_position = np.random.choice(atom_array).ref_pos
            if len(avg_ligand_position.shape) == 1:
                avg_ligand_position = np.expand_dims(avg_ligand_position, 0)
        else:
            residue_chosen = np.random.choice(non_protein_residues)
            residue_indices = np.where(atom_array.res_name == residue_chosen)[0]
            residue_coords = atom_array.ref_pos[residue_indices]
            # avg coord
            avg_ligand_position = np.expand_dims(np.mean(residue_coords, axis=0), 0)
        # distance matrix
        dist_matrix = cdist(atom_array.ref_pos, avg_ligand_position)
        # mask residues that are close to the ligand
        cutoff = np.percentile(dist_matrix, 100 * t)
        masked_res_indices = np.where(dist_matrix < cutoff)[0]
        masked_sym_uid_hashes = self.sym_uid_hashes[masked_res_indices]
        masked_sym_uuid_hashes = np.unique(masked_sym_uid_hashes)
        masked_sym_uuid_indices = np.where(np.isin(self.sym_uid_hashes, masked_sym_uuid_hashes))[0]
        # return np.intersect1d(masked_sym_uuid_indices, self.is_protein_indices)
        return np.intersect1d(masked_sym_uuid_indices, self.is_molecule_indices)

    def preprocess_atom_array(self, atom_array):
        self.ref_space_uid = atom_array.ref_space_uid
        self.entity_mol_id = atom_array.entity_mol_id
        self.res_id = atom_array.res_id
        # Stupid hash for unique values
        self.sym_uid_hashes = self.entity_mol_id * 10000 + self.res_id
        self.is_protein_mask = atom_array.is_protein.astype("bool")
        self.is_rna_mask = atom_array.is_rna.astype("bool")
        self.is_protein_indices = np.where(self.is_protein_mask)[0]
        self.is_rna_indices = np.where(self.is_rna_mask)[0]
        if self.mask_na:
            self.is_molecule_indices = np.union1d(self.is_protein_indices, self.is_rna_indices)
        else:
            self.is_molecule_indices = self.is_protein_indices
        self.sym_protein_uid_hashes = self.sym_uid_hashes[self.is_protein_mask]
        self.sym_uuid_hashes = np.unique(self.sym_protein_uid_hashes)

    def mask_ref_information(
        self,
        atom_array,
        masked_res_indices,
        backbone_atoms=None,
    ):
        """
        Various ref information is used in the atom attention encoder when
        computing s_init. As such, here we mask the ref information so it
        cannot be used by the model downstream to memorize sequence information.

        Args:
            atom_array (dict): The atom array.

        Returns:
            dict: A new bioassembly dictionary with ref information of masked backbone
                  atoms changed to that of the UNK residue.

        """
        backbone_atoms = self.backbone_atoms
        if self.mask_na:
            backbone_atoms += self.na_backbone_atoms
        backbone_atom_indices = np.flatnonzero(np.isin(atom_array.atom_name, backbone_atoms))
        masked_backbone_indices = np.intersect1d(backbone_atom_indices, masked_res_indices)

        mask_idx = np.isin(np.arange(len(atom_array)), masked_backbone_indices)

        # This can happen if we're doing cogen and t is close to 0
        if not mask_idx.any():
            return atom_array

        masked_atom_array = atom_array[mask_idx]
        _mask_ref = _get_mask_ref()
        ref_pos, ref_mask, ref_charge = [], [], []
        for atom in atom_array[mask_idx]:
            atom_sub_idx = _mask_ref["atom_map"][atom.atom_name]
            ref_pos.append(_mask_ref["pos"][atom_sub_idx])
            ref_mask.append(_mask_ref["mask"][atom_sub_idx])
            ref_charge.append(_mask_ref["charge"][atom_sub_idx])

        ref_pos = np.array(ref_pos)
        if self._transform_masked_ref_pos:
            trfmd_ref_pos = []
            for ref_space_uid in np.unique(masked_atom_array.ref_space_uid):
                trfmd_ref_pos.append(
                    random_transform(
                        ref_pos[masked_atom_array.ref_space_uid == ref_space_uid],
                        apply_augmentation=self._ref_pos_augment,
                        centralize=True,
                    )
                )

            ref_pos = np.concatenate(trfmd_ref_pos, axis=0)

        ref_pos_all = atom_array.ref_pos
        ref_charge_all = atom_array.ref_charge
        ref_mask_all = atom_array.ref_mask

        ref_pos_all[mask_idx] = ref_pos
        ref_charge_all[mask_idx] = np.array(ref_charge).astype(int)
        ref_mask_all[mask_idx] = np.array(ref_mask).astype(int)

        atom_array.set_annotation("ref_pos", ref_pos_all)
        atom_array.set_annotation("ref_charge", ref_charge_all)
        atom_array.set_annotation("ref_mask", ref_mask_all)

        return atom_array

    def mask_side_chains(
        self,
        atom_array,
        masked_res_indices,
        backbone_atoms=None,
    ):
        """
        Apply a placeholder mask to the bioassembly dictionary.

        Args:
            atom_array (biotite.AtomArray): The atom array.
            mask (np.ndarray): Boolean mask for the atom_array (True for atoms to keep, False to mask).

        Returns:
            dict: A new bioassembly dictionary with placeholders applied.

            @NOTE: What is really changed is the side chain (masked, removed)
        """
        backbone_atoms = self.backbone_atoms
        if self.mask_na:
            backbone_atoms += self.na_backbone_atoms
        non_backbone_atom_indices = np.where(~np.isin(atom_array.atom_name, backbone_atoms))[0]
        masked_side_chain_indices = np.intersect1d(non_backbone_atom_indices, masked_res_indices)
        # non_backbone_atom_indices may also include non protein non backbone atoms
        masked_side_chain_non_protein_indices = np.where(
            ~np.isin(masked_side_chain_indices, self.is_protein_indices)
        )[0]
        side_chain_mask = np.ones(len(atom_array), dtype=bool)
        side_chain_mask[masked_side_chain_indices] = False

        # Mask the atom array. Set only the cano resname as its used in the data
        # leave the resname as the true resname
        atom_array.set_annotation("clean_cano_seq_resname", np.copy(atom_array.cano_seq_resname))
        atom_array.cano_seq_resname[masked_res_indices] = MASK_RESNAME
        atom_array = atom_array[side_chain_mask]  # Mask the atom array
        atom_array = AddAtomArrayAnnot.add_distogram_rep_atom_mask_taskmanager(atom_array)
        return atom_array

    def sample_task_and_time(self):
        """_summary_

        Args:
            data_d (_type_): _description_
            masked_data_d (_type_): _description_

            keys that altered:
            - ref_atom_name_chars
            - restype
            - ref_mask
        """
        task = self._choose_task()
        match task:
            case "folding":
                t = 0
            case "unconditional_structure_no_cogen":
                t = 1
            case "cogen" | "unconditional_structure" | "inverse_folding":
                t = np.random.rand()
            case _:
                raise ValueError("Invalid task")

        return task, t
