import copy
import os
import functools
import re
from collections import defaultdict
from typing import Mapping, Sequence

import biotite.structure as struc
import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile

from rdkit import Chem

from oxtal.data.constants import STD_RESIDUES, COORDINATION_DONORS, COORDINATION_ACCEPTORS


def remove_numbers(s: str) -> str:
    """
    Remove numbers from a string.

    Args:
        s (str): input string

    Returns:
        str: a string with numbers removed.
    """
    return re.sub(r"\d+", "", s)


def get_inter_residue_bonds(atom_array: AtomArray) -> np.ndarray:
    """get inter residue bonds by checking chain_id and res_id

    Args:
        atom_array (AtomArray): Biotite AtomArray, must have chain_id and res_id

    Returns:
        np.ndarray: inter residue bonds, shape = (n,2)
    """
    if atom_array.bonds is None:
        return []
    idx_i = atom_array.bonds._bonds[:, 0]
    idx_j = atom_array.bonds._bonds[:, 1]
    chain_id_diff = atom_array.chain_id[idx_i] != atom_array.chain_id[idx_j]
    res_id_diff = atom_array.res_id[idx_i] != atom_array.res_id[idx_j]
    diff_mask = chain_id_diff | res_id_diff
    inter_residue_bonds = atom_array.bonds._bonds[diff_mask]
    inter_residue_bonds = inter_residue_bonds[:, :2]  # remove bond type
    return inter_residue_bonds


def get_starts_by(atom_array: AtomArray, by_annot: str, add_exclusive_stop=False) -> np.ndarray:
    """get start indices by given annotation in an AtomArray

    Args:
        atom_array (AtomArray): Biotite AtomArray
        by_annot (str): annotation to group by, eg: 'chain_id', 'res_id', 'res_name'
        add_exclusive_stop (bool, optional): add exclusive stop (len(atom_array)). Defaults to False.

    Returns:
        np.ndarray: start indices of each group, shape = (n,), eg: [0, 10, 20, 30, 40]
    """
    annot = getattr(atom_array, by_annot)
    # If annotation change, a new start
    annot_change_mask = annot[1:] != annot[:-1]

    # Convert mask to indices
    # Add 1, to shift the indices from the end of a residue
    # to the start of a new residue
    starts = np.where(annot_change_mask)[0] + 1

    # The first start is not included yet -> Insert '[0]'
    if add_exclusive_stop:
        return np.concatenate(([0], starts, [atom_array.array_length()]))
    else:
        return np.concatenate(([0], starts))


def atom_select(atom_array: AtomArray, select_dict: dict, as_mask=False) -> np.ndarray:
    """return index of atom_array that match select_dict

    Args:
        atom_array (AtomArray): Biotite AtomArray
        select_dict (dict): select dict, eg: {'element': 'C'}
        as_mask (bool, optional): return mask of atom_array. Defaults to False.

    Returns:
        np.ndarray: index of atom_array that match select_dict
    """
    mask = np.ones(len(atom_array), dtype=bool)
    for k, v in select_dict.items():
        mask = mask & (getattr(atom_array, k) == v)
    if as_mask:
        return mask
    else:
        return np.where(mask)[0]


def get_ligand_polymer_bond_mask(atom_array: AtomArray, lig_include_ions=False) -> np.ndarray:
    """
    Ref AlphaFold3 SI Chapter 3.7.1.
    Get bonds between the bonded ligand and its parent chain.

    Args:
        atom_array (AtomArray): biotite atom array object.
        lig_include_ions (bool): whether to include ions in the ligand.

    Returns:
        np.ndarray: bond records between the bonded ligand and its parent chain.
                    e.g. np.array([[atom1, atom2, bond_order]...])
    """
    if not lig_include_ions:
        # bonded ligand exclude ions
        unique_chain_id, counts = np.unique(atom_array.label_asym_id, return_counts=True)
        chain_id_to_count_map = dict(zip(unique_chain_id, counts))
        ions_mask = np.array(
            [
                chain_id_to_count_map[label_asym_id] == 1
                for label_asym_id in atom_array.label_asym_id
            ]
        )

        lig_mask = (atom_array.mol_type == "ligand") & ~ions_mask
    else:
        lig_mask = atom_array.mol_type == "ligand"

    # identify polymer by mol_type (protein, rna, dna, ligand)
    polymer_mask = np.isin(atom_array.mol_type, ["protein", "rna", "dna"])

    idx_i = atom_array.bonds._bonds[:, 0]
    idx_j = atom_array.bonds._bonds[:, 1]

    lig_polymer_bond_indices = np.where(
        (lig_mask[idx_i] & polymer_mask[idx_j]) | (lig_mask[idx_j] & polymer_mask[idx_i])
    )[0]
    if lig_polymer_bond_indices.size == 0:
        # no ligand-polymer bonds
        lig_polymer_bonds = np.empty((0, 3)).astype(int)
    else:
        lig_polymer_bonds = atom_array.bonds._bonds[
            lig_polymer_bond_indices
        ]  # np.array([[atom1, atom2, bond_order]...])
    return lig_polymer_bonds


@functools.lru_cache
def parse_pdb_cluster_file_to_dict(
    cluster_file: str, remove_uniprot: bool = True
) -> dict[str, tuple]:
    """parse PDB cluster file, and return a pandas dataframe
    example cluster file:
    https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt

    Args:
        cluster_file (str): cluster_file path
    Returns:
        dict(str, tuple(str, str)): {pdb_id}_{entity_id} --> [cluster_id, cluster_size]
    """
    pdb_cluster_dict = {}
    with open(cluster_file) as f:
        for line in f:
            pdb_clusters = []
            for ids in line.strip().split():
                if remove_uniprot:
                    if ids.startswith("AF_") or ids.startswith("MA_"):
                        continue
                pdb_clusters.append(ids)
            cluster_size = len(pdb_clusters)
            if cluster_size == 0:
                continue
            # use first member as cluster id.
            cluster_id = f"pdb_cluster_{pdb_clusters[0]}"
            for ids in pdb_clusters:
                pdb_cluster_dict[ids.lower()] = (cluster_id, cluster_size)
    return pdb_cluster_dict

def get_clean_data(atom_array: AtomArray) -> AtomArray:
    """
    Removes unresolved atoms from the AtomArray.

    Args:
        atom_array (AtomArray): The input AtomArray containing atoms.

    Returns:
        AtomArray: A new AtomArray with unresolved atoms removed.
    """
    atom_array_wo_unresol = atom_array.copy()
    atom_array_wo_unresol = atom_array[atom_array.is_resolved]
    return atom_array_wo_unresol


def save_atoms_to_cif(
    output_cif_file: str,
    atom_array: AtomArray,
    entity_poly_type: dict[str, str],
    pdb_id: str,
) -> None:
    """
    Save atom array data to a CIF file.

    Args:
        output_cif_file (str): The output path for saving the atom array in CIF format.
        atom_array (AtomArray): The atom array to be saved.
        entity_poly_type: The entity poly type information.
        pdb_id: The PDB ID for the entry.
    """
    cifwriter = CIFWriter(atom_array, entity_poly_type)
    cifwriter.save_to_cif(
        output_path=output_cif_file,
        entry_id=pdb_id,
        include_bonds=True,
    )


def save_atoms_to_pdb(
    output_pdb_file: str,
    atom_array: AtomArray,
    entity_poly_type: dict[str, str],
    pdb_id: str,
) -> None:
    """
    Save atom array data to a PDB file.

    Args:
        output_cif_file (str): The output path for saving the atom array in CIF format.
        atom_array (AtomArray): The atom array to be saved.
        entity_poly_type: The entity poly type information.
        pdb_id: The PDB ID for the entry.
    """
    pdb_file = PDBFile()
    atom_array.chain_id = [a[0] for a in atom_array.label_asym_id]
    pdb_file.set_structure(atom_array)
    pdb_file.write(output_pdb_file)


def save_structure_pdb(
    atom_array: AtomArray,
    pred_coordinate: torch.Tensor,
    output_fpath: str,
    entity_poly_type: dict[str, str],
    pdb_id: str,
):
    pred_atom_array = copy.deepcopy(atom_array)
    pred_pose = pred_coordinate.cpu().numpy()
    pred_atom_array.coord = pred_pose
    save_atoms_to_pdb(
        output_fpath,
        pred_atom_array,
        entity_poly_type,
        pdb_id,
    )
    # save pred coordinates wo unresolved atoms
    if hasattr(atom_array, "is_resolved"):
        pred_atom_array_wo_unresol = get_clean_data(pred_atom_array)
        save_atoms_to_pdb(
            output_fpath.replace(".pdb", "_wounresol.pdb"),
            pred_atom_array_wo_unresol,
            entity_poly_type,
            pdb_id,
        )


def save_structure_cif(
    atom_array: AtomArray,
    pred_coordinate: torch.Tensor,
    output_fpath: str,
    entity_poly_type: dict[str, str],
    pdb_id: str,
):
    """
    Save the predicted structure to a CIF file.

    Args:
        atom_array (AtomArray): The original AtomArray containing the structure.
        pred_coordinate (torch.Tensor): The predicted coordinates for the structure.
        output_fpath (str): The output file path for saving the CIF file.
        entity_poly_type (dict[str, str]): The entity poly type information.
        pdb_id (str): The PDB ID for the entry.
    """
    pred_atom_array = copy.deepcopy(atom_array)
    pred_pose = pred_coordinate.cpu().numpy()
    pred_atom_array.coord = pred_pose
    save_atoms_to_cif(
        output_fpath,
        pred_atom_array,
        entity_poly_type,
        pdb_id,
    )
    # save pred coordinates wo unresolved atoms
    if hasattr(atom_array, "is_resolved"):
        pred_atom_array_wo_unresol = get_clean_data(pred_atom_array)
        save_atoms_to_cif(
            output_fpath.replace(".cif", "_wounresol.cif"),
            pred_atom_array_wo_unresol,
            entity_poly_type,
            pdb_id,
        )


class CIFWriter:
    """
    Write AtomArray to cif.
    """

    def __init__(self, atom_array: AtomArray, entity_poly_type: dict[str, str] = None):
        """
        Args:
            atom_array (AtomArray): Biotite AtomArray object.
            entity_poly_type (dict[str, str], optional): A dict of label_entity_id to entity_poly_type. Defaults to None.
                                                         If None, "the entity_poly" and "entity_poly_seq" will not be written to the cif.
        """
        self.atom_array = atom_array
        self.entity_poly_type = entity_poly_type

    def _get_entity_block(self):
        if self.entity_poly_type is None:
            return {}
        entity_ids_in_atom_array = np.sort(np.unique(self.atom_array.label_entity_id))
        entity_block_dict = defaultdict(list)
        for entity_id in entity_ids_in_atom_array:
            if entity_id not in self.entity_poly_type:
                entity_type = "non-polymer"
            else:
                entity_type = "polymer"
            entity_block_dict["id"].append(entity_id)
            entity_block_dict["pdbx_description"].append(".")
            entity_block_dict["type"].append(entity_type)
        return pdbx.CIFCategory(entity_block_dict)
    
    def _get_entity_poly_and_entity_poly_seq_block(self):
        entity_poly = defaultdict(list)
        for entity_id, entity_type in self.entity_poly_type.items():
            label_asym_ids = np.unique(
                self.atom_array.label_asym_id[self.atom_array.label_entity_id == entity_id]
            )
            label_asym_ids_str = ",".join(label_asym_ids)

            if label_asym_ids_str == "":
                # The entity not in current atom_array
                continue

            entity_poly["entity_id"].append(entity_id)
            entity_poly["pdbx_strand_id"].append(label_asym_ids_str)
            entity_poly["type"].append(entity_type)

        if not entity_poly:
            return {}

        entity_poly_seq = defaultdict(list)
        for entity_id, label_asym_ids_str in zip(
            entity_poly["entity_id"], entity_poly["pdbx_strand_id"]
        ):
            first_label_asym_id = label_asym_ids_str.split(",")[0]
            first_asym_chain = self.atom_array[
                self.atom_array.label_asym_id == first_label_asym_id
            ]
            chain_starts = struc.get_chain_starts(first_asym_chain, add_exclusive_stop=True)
            asym_chain = first_asym_chain[
                chain_starts[0] : chain_starts[1]
            ]  # ensure the asym chain is a single chain

            res_starts = struc.get_residue_starts(asym_chain, add_exclusive_stop=False)
            asym_chain_entity_id = asym_chain[res_starts].label_entity_id.tolist()
            asym_chain_hetero = ["n" if not i else "y" for i in asym_chain[res_starts].hetero]
            asym_chain_res_name = asym_chain[res_starts].res_name.tolist()
            asym_chain_res_id = asym_chain[res_starts].res_id.tolist()

            entity_poly_seq["entity_id"].extend(asym_chain_entity_id)
            entity_poly_seq["hetero"].extend(asym_chain_hetero)
            entity_poly_seq["mon_id"].extend(asym_chain_res_name)
            entity_poly_seq["num"].extend(asym_chain_res_id)

        block_dict = {
            "entity_poly": pdbx.CIFCategory(entity_poly),
            "entity_poly_seq": pdbx.CIFCategory(entity_poly_seq),
        }
        return block_dict

    def save_to_cif(self, output_path: str, entry_id: str = None, include_bonds: bool = False):
        """
        Save AtomArray to cif.

        Args:
            output_path (str): Output path of cif file.
            entry_id (str, optional): The value of "_entry.id" in cif. Defaults to None.
                                      If None, the entry_id will be the basename of output_path (without ".cif" extension).
            include_bonds (bool, optional): Whether to include  bonds in the cif. Defaults to False.
                                            If set to True and `array` has associated ``bonds`` , the
                                            intra-residue bonds will be written into the ``chem_comp_bond``
                                            category.
                                            Inter-residue bonds will be written into the ``struct_conn``
                                            independent of this parameter.

        """
        if entry_id is None:
            entry_id = os.path.basename(output_path).replace(".cif", "")

        block_dict = {"entry": pdbx.CIFCategory({"id": entry_id})}
        if self.entity_poly_type is not None:
            block_dict["entity"] = self._get_entity_block()
            block_dict.update(self._get_entity_poly_and_entity_poly_seq_block())

        block = pdbx.CIFBlock(block_dict)
        cif = pdbx.CIFFile(
            {os.path.basename(output_path).replace(".cif", "") + "_predicted_by_oxtal": block}
        )
        pdbx.set_structure(cif, self.atom_array, include_bonds=include_bonds)
        block = cif.block
        atom_site = block.get("atom_site")
        atom_site["label_entity_id"] = self.atom_array.label_entity_id
        cif.write(output_path)


def make_dummy_feature(
    features_dict: Mapping[str, torch.Tensor],
    dummy_feats: Sequence = ["msa"],
) -> dict[str, torch.Tensor]:
    num_token = features_dict["token_index"].shape[0]
    num_atom = features_dict["atom_to_token_idx"].shape[0]
    num_msa = 1
    num_templ = 4
    num_pockets = 30
    feat_shape, _ = get_data_shape_dict(
        num_token=num_token,
        num_atom=num_atom,
        num_msa=num_msa,
        num_templ=num_templ,
        num_pocket=num_pockets,
    )
    for feat_name in dummy_feats:
        if feat_name not in ["msa", "template"]:
            cur_feat_shape = feat_shape[feat_name]
            features_dict[feat_name] = torch.zeros(cur_feat_shape)
    if "msa" in dummy_feats:
        # features_dict["msa"] = features_dict["restype"].unsqueeze(0)
        features_dict["msa"] = torch.nonzero(features_dict["restype"])[:, 1].unsqueeze(0)
        assert features_dict["msa"].shape == feat_shape["msa"]
        features_dict["has_deletion"] = torch.zeros(feat_shape["has_deletion"])
        features_dict["deletion_value"] = torch.zeros(feat_shape["deletion_value"])
        features_dict["profile"] = features_dict["restype"]
        assert (
            features_dict["profile"].shape == feat_shape["profile"]
        ), f"dict shape: {features_dict['profile'].shape, feat_shape['profile']}"
        features_dict["deletion_mean"] = torch.zeros(feat_shape["deletion_mean"])
        for key in [
            "prot_pair_num_alignments",
            "prot_unpair_num_alignments",
            "rna_pair_num_alignments",
            "rna_unpair_num_alignments",
        ]:
            features_dict[key] = torch.tensor(0, dtype=torch.int32)

    if "template" in dummy_feats:
        features_dict["template_restype"] = torch.ones(feat_shape["template_restype"]) * 31  # gap
        features_dict["template_all_atom_mask"] = torch.zeros(feat_shape["template_all_atom_mask"])
        features_dict["template_all_atom_positions"] = torch.zeros(
            feat_shape["template_all_atom_positions"]
        )
    return features_dict


def data_type_transform(
    feat_or_label_dict: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], AtomArray]:
    for key, value in feat_or_label_dict.items():
        if key in IntDataList:
            feat_or_label_dict[key] = value.to(torch.long)

    return feat_or_label_dict


# List of "index" or "type" data
# Their data type should be int
IntDataList = [
    "residue_index",
    "token_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "ref_space_uid",
    "template_restype",
    "atom_to_token_idx",
    "atom_to_tokatom_idx",
    "frame_atom_index",
    "msa",
    "entity_mol_id",
    "mol_id",
    "mol_atom_index",
]


# shape of the data
def get_data_shape_dict(num_token, num_atom, num_msa, num_templ, num_pocket):
    """
    Generate a dictionary containing the shapes of all data.

    Args:
        num_token (int): Number of tokens.
        num_atom (int): Number of atoms.
        num_msa (int): Number of MSA sequences.
        num_templ (int): Number of templates.
        num_pocket (int): Number of pockets to the same interested ligand.

    Returns:
        dict: A dictionary containing the shapes of all data.
    """
    # Features in AlphaFold3 SI Table5
    feat = {
        # Token features
        "residue_index": (num_token,),
        "token_index": (num_token,),
        "asym_id": (num_token,),
        "entity_id": (num_token,),
        "sym_id": (num_token,),
        "restype": (num_token, len(STD_RESIDUES) + 1),
        # chain permutation features
        "entity_mol_id": (num_atom,),
        "mol_id": (num_atom,),
        "mol_atom_index": (num_atom,),
        # Reference features
        "ref_pos": (num_atom, 3),
        "ref_mask": (num_atom,),
        "ref_element": (num_atom, 128),  # note: 128 elem in the paper
        "ref_charge": (num_atom,),
        "ref_atom_name_chars": (num_atom, 4, 64),
        "ref_space_uid": (num_atom,),
        # Msa features
        # "msa": (num_msa, num_token, 32),
        "msa": (num_msa, num_token),
        "has_deletion": (num_msa, num_token),
        "deletion_value": (num_msa, num_token),
        "profile": (num_token, len(STD_RESIDUES) + 1),
        "deletion_mean": (num_token,),
        # Template features
        "template_restype": (num_templ, num_token),
        "template_all_atom_mask": (num_templ, num_token, 37),
        "template_all_atom_positions": (num_templ, num_token, 37, 3),
        "template_pseudo_beta_mask": (num_templ, num_token),
        "template_backbone_frame_mask": (num_templ, num_token),
        "template_distogram": (num_templ, num_token, num_token, 39),
        "template_unit_vector": (num_templ, num_token, num_token, 3),
        # Bond features
        "token_bonds": (num_token, num_token),
    }

    # Extra features needed
    extra_feat = {
        # Input features
        "atom_to_token_idx": (num_atom,),  # after crop
        "atom_to_tokatom_idx": (num_atom,),  # after crop
        "pae_rep_atom_mask": (num_atom,),  # same as "pae_rep_atom_mask" in label_dict
        "is_distillation": (1,),
    }

    # Label
    label = {
        "coordinate": (num_atom, 3),
        "coordinate_mask": (num_atom,),
        # "centre_atom_mask": (num_atom,),
        # "centre_centre_distance": (num_token, num_token),
        # "centre_centre_distance_mask": (num_token, num_token),
        "distogram_rep_atom_mask": (num_atom,),
        "pae_rep_atom_mask": (num_atom,),
        "plddt_m_rep_atom_mask": (num_atom,),
        "modified_res_mask": (num_atom,),
        "bond_mask": (num_atom, num_atom),
        "is_protein": (num_atom,),  # Atom level, not token level
        "is_rna": (num_atom,),
        "is_dna": (num_atom,),
        "is_ligand": (num_atom,),
        "has_frame": (num_token,),  # move to input_feature_dict?
        "frame_atom_index": (num_token, 3),  # atom index after crop
        "resolution": (1,),
        # Metrics
        "interested_ligand_mask": (
            num_pocket,
            num_atom,
        ),
        "pocket_mask": (
            num_pocket,
            num_atom,
        ),
    }

    # Merged
    all_feat = {**feat, **extra_feat}
    return all_feat, label

def lattice_from_params(lengths, angles):
    """Calculate the lattice transformation matrix from unit cell lengths and angles.

    Args:
        lengths (np.array): Unit cell lengths a, b, c.
        angles (np.array): Unit cell angles alpha, beta, gamma.

    Returns:
        np.array: Lattice transformation matrix.
    """
    alpha, beta, gamma = np.radians(angles)
    a, b, c = lengths
    volume = (
        a
        * b
        * c
        * np.sqrt(
            1
            - np.cos(alpha) ** 2
            - np.cos(beta) ** 2
            - np.cos(gamma) ** 2
            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
    )
    matrix = np.zeros((3, 3))
    matrix[0] = [a, 0, 0]
    matrix[1] = [b * np.cos(gamma), b * np.sin(gamma), 0]
    matrix[2] = [
        c * np.cos(beta),
        c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
        volume / (a * b * np.sin(gamma)),
    ]
    return matrix

def preprocess_smiles(smiles: str, return_mol: bool = False) -> str:  # pylint: disable=method-hidden
    """Very small normalisation routine so that *every* component can be
    converted by RDKit if the user needs that later on."""
    smiles = smiles.strip()
    # Project‑specific normalisation tweaks – adjust or remove if not desired.
    smiles = (
        smiles.replace("C#O", "[O+]#[C-]")  # neutralise *C#O*
        .replace("([2H])", "")
        .replace("[2H]", "")
    )

    # canonicalize the SMILES
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol = make_coordinate_bonds(mol)
        if mol is None:
            raise ValueError("MolFromSmiles returned None")
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise ValueError(f"RDKit parsing error: {exc}")
    if return_mol:
        return mol
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True) # if mol else smi

def make_coordinate_bonds(mol):
    pt  = Chem.GetPeriodicTable()
    rw  = Chem.RWMol(mol)
    rw.UpdatePropertyCache(strict=False)

    for bond in tuple(rw.GetBonds()):                 # work on a copy
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        for donor, acc in ((a, b), (b, a)):
            if (donor.GetAtomicNum() in COORDINATION_DONORS and
                acc.GetAtomicNum()   in COORDINATION_ACCEPTORS and
                bond.GetBondType()   == Chem.BondType.SINGLE and
                donor.GetExplicitValence() > pt.GetDefaultValence(donor.GetAtomicNum())):

                rw.RemoveBond(donor.GetIdx(), acc.GetIdx())
                rw.AddBond(donor.GetIdx(), acc.GetIdx(), Chem.BondType.DATIVE)
                break
    return rw.GetMol()

def molblock_to_mol(molblock: str, remove_hydrogens: bool = True) -> Chem.Mol | None:
    """Wrapper around :pyfunc:`rdkit.Chem.MolFromMolBlock` with strict flags."""

    mol = Chem.MolFromMolBlock(
            molblock,
            sanitize=False,     # <- don’t let RDKit gag yet
            removeHs=False,
            strictParsing=False # tolerate bad valence tables, etc.
    )
    mol = make_coordinate_bonds(mol)
    Chem.SanitizeMol(mol, catchErrors=True)
    if remove_hydrogens:
        mol = Chem.RemoveHs(mol)
    return mol