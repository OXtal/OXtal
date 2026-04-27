import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray

from oxtal.data.parser import DistillationMMCIFParser, MMCIFParser, CIFCrystalParser
from oxtal.data.tokenizer import AtomArrayTokenizer
from oxtal.utils.file_io import load_gzip_pickle

torch.multiprocessing.set_sharing_strategy("file_system")


class DataPipeline:
    """
    DataPipeline class provides static methods to handle various data processing tasks related to bioassembly structures.
    """

    @staticmethod
    def get_data_from_mmcif(
        mmcif: Union[str, Path],
        pdb_cluster_file: Union[str, Path, None] = None,
        dataset: str = "WeightedPDB",
        crystal_only: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Get raw data from mmcif with tokenizer and a list of chains and interfaces for sampling.

        Args:
            mmcif (Union[str, Path]): The raw mmcif file.
            pdb_cluster_file (Union[str, Path, None], optional): Cluster info txt file. Defaults to None.
            dataset (str, optional): The dataset type, either "WeightedPDB" or "Distillation". Defaults to "WeightedPDB".

        Returns:
            tuple[list[dict[str, Any]], dict[str, Any]]:
                sample_indices_list (list[dict[str, Any]]): The sample indices list (each one is a chain or an interface).
                bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array, and token_array.
        """
        try:
            if dataset == "WeightedPDB":
                parser = MMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_bioassembly()
            elif dataset == "Distillation":
                parser = DistillationMMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_structure_dict()
            elif dataset == "CCDC":
                parser = CIFCrystalParser(identifier=mmcif)
                bioassembly_dict = parser.get_bioassembly()
            else:
                raise NotImplementedError(
                    'Unsupported "dataset", please input either "WeightedPDB" or "Distillation".'
                )

            sample_indices_list = parser.make_indices(
                bioassembly_dict=bioassembly_dict, pdb_cluster_file=pdb_cluster_file
            )
            if len(sample_indices_list) == 0:
                # empty indices and AtomArray
                return [], bioassembly_dict

            atom_array = bioassembly_dict["atom_array"]
            atom_array.set_annotation(
                "resolution", [parser.resolution] * len(atom_array)
            )

            tokenizer = AtomArrayTokenizer(atom_array, crystal_only=crystal_only)
            token_array = tokenizer.get_token_array()
            bioassembly_dict["msa_features"] = None
            bioassembly_dict["template_features"] = None

            bioassembly_dict["token_array"] = token_array
            return sample_indices_list, bioassembly_dict

        except Exception as e:
            logging.warning("Gen data failed for %s due to %s", mmcif, e)
            return [], {}

    @staticmethod
    def get_label_entity_id_to_asym_id_int(atom_array: AtomArray) -> dict[str, int]:
        """
        Get a dictionary that associates each label_entity_id with its corresponding asym_id_int.

        Args:
            atom_array (AtomArray): AtomArray object

        Returns:
            dict[str, int]: label_entity_id to its asym_id_int
        """
        entity_to_asym_id = defaultdict(set)
        for atom in atom_array:
            entity_id = atom.label_entity_id
            entity_to_asym_id[entity_id].add(atom.asym_id_int)
        return entity_to_asym_id

    @staticmethod
    def get_data_bioassembly(
        bioassembly_dict_fpath: Union[str, Path],
    ) -> dict[str, Any]:
        """
        Get the bioassembly dict.

        Args:
            bioassembly_dict_fpath (Union[str, Path]): The path to the bioassembly dictionary file.

        Returns:
            dict[str, Any]: The bioassembly dict with sequence, atom_array and token_array.

        Raises:
            AssertionError: If the bioassembly dictionary file does not exist.
        """
        assert os.path.exists(bioassembly_dict_fpath), f"File not exists {bioassembly_dict_fpath}"
        bioassembly_dict = load_gzip_pickle(bioassembly_dict_fpath)

        return bioassembly_dict

    @staticmethod
    def _map_ref_chain(one_sample: pd.Series, bioassembly_dict: dict[str, Any]) -> list[int]:
        """
        Map the chain or interface chain_x_id to the reference chain asym_id.

        Args:
            one_sample (pd.Series): A dict of one chain or interface from indices list.
            bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array and token_array.

        Returns:
            list[int]: A list of asym_id_lnt of the chosen chain or interface, length 1 or 2.
        """
        atom_array = bioassembly_dict["atom_array"]
        ref_chain_indices = []
        for chain_id_field in ["chain_1_id", "chain_2_id"]:
            chain_id = one_sample[chain_id_field]
            assert np.isin(
                chain_id, np.unique(atom_array.chain_id)
            ), f"PDB {bioassembly_dict['pdb_id']} {chain_id_field}:{chain_id} not in atom_array"
            chain_asym_id = atom_array[atom_array.chain_id == chain_id].asym_id_int[0]
            ref_chain_indices.append(chain_asym_id)
            if one_sample["type"] == "chain":
                break
        return ref_chain_indices

    @staticmethod
    def save_atoms_to_cif(
        output_cif_file: str, atom_array: AtomArray, include_bonds: bool = False
    ) -> None:
        """
        Save atom array data to a CIF file.

        Args:
            output_cif_file (str): The output path for saving atom array in cif
            atom_array (AtomArray): The atom array to be saved
            include_bonds (bool): Whether to include bond information in the CIF file. Default is False.

        """
        strucio.save_structure(
            file_path=output_cif_file,
            array=atom_array,
            data_block=os.path.basename(output_cif_file).replace(".cif", ""),
            include_bonds=include_bonds,
        )
