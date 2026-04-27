import copy
import json
import logging
import time
import traceback
import warnings
from typing import Any, List, Mapping, Optional, Tuple, Union

import hydra
import numpy as np
import omegaconf
import torch
from biotite.structure import AtomArray
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from oxtal.data.constants import (
    MASK_STD_RESIDUES,
    PRO_STD_RESIDUES,
    PRO_STD_RESIDUES_VALS_SET,
)
from oxtal.data.json_to_feature import SampleDictToFeatures
from oxtal.data.task_manager import TaskManager
from oxtal.data.utils import data_type_transform, make_dummy_feature
from oxtal.utils.torch_utils import dict_to_tensor

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


def get_inference_dataloader(
    fabric, configs: Any, num_eval_seeds: Optional[Union[int, List[int]]] = None
) -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the InferenceDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.

    Returns:
        A DataLoader object configured for inference.
    """
    inference_dataset = InferenceDataset(
        input_json_path=configs.input_json_path,
        dump_dir=configs.dump_dir,
        use_msa=configs.use_msa,
        crystal_only=configs.crystal_only,
        task_manager=hydra.utils.instantiate(configs.task_manager),
    )

    if num_eval_seeds is not None:
        datasets_to_concat = []

        list_types = (list, omegaconf.listconfig.ListConfig)
        seed_iter = (
            num_eval_seeds if isinstance(num_eval_seeds, list_types) else range(num_eval_seeds)
        )

        for seed in seed_iter:
            this_dataset = copy.deepcopy(inference_dataset)
            this_dataset.set_seed(seed)
            datasets_to_concat.append(this_dataset)

        inference_dataset = torch.utils.data.ConcatDataset(datasets_to_concat)

    sampler = None
    if fabric is not None:
        sampler = DistributedSampler(
            dataset=inference_dataset,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=False,
        )

    dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch,
        num_workers=configs.num_workers,
    )
    return dataloader


def build_inference_features(
    sample2feat: SampleDictToFeatures,
    use_msa: bool,
    atom_array: Optional[AtomArray] = None,
    bb_only: bool = False,
    crystal_only: bool = False,
) -> Tuple[dict, AtomArray, dict]:
    """
    Given an already initialized sample2feat builds inference features. Can
    take an optional atom_array, in which case this method does not remake
    the atom array in sample2feat and instead reuses the atom_array passed in.
    This is used during inference when a residue is changed by the discrete
    diffusion model.

    Args:
        sample2feat (SampleDictToFeatures): Object which makes base of feature dict
        use_msa (bool): Whether or not we're using MSA features
        atom_array (Optional[biotite.AtomArray]): If specified, use this atom array when getting
                                                  features from sample2feat instead of generating
                                                  it from scratch.

    Returns:
        Tuple[dict, biotite.AtomArray, dict]: First dict is features, then the atom array,
                                              then a dict which tracks how much time the
                                              featurizing took.
    """
    t0 = time.time()
    features_dict, atom_array, token_array = sample2feat.get_feature_dict(
        atom_array, called_during_generation=atom_array is not None, bb_only=bb_only, crystal_only=crystal_only
    )

    features_dict["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    entity_poly_type = sample2feat.entity_poly_type
    t1 = time.time()

    msa_features = {}

    # Make dummy features for not implemented features
    dummy_feats = ["template"]
    if len(msa_features) == 0:
        dummy_feats.append("msa")
    else:
        msa_features = dict_to_tensor(msa_features)
        features_dict.update(msa_features)
    features_dict = make_dummy_feature(
        features_dict=features_dict,
        dummy_feats=dummy_feats,
    )

    # Transform to right data type
    feat = data_type_transform(feat_or_label_dict=features_dict)

    masked_residues = PRO_STD_RESIDUES | MASK_STD_RESIDUES
    cond = lambda x, i: (
        (x in masked_residues)
        and (feat["true_restype_id"][i] in masked_residues)
        and (token_array[i].value in PRO_STD_RESIDUES_VALS_SET)
    )

    feat["atom_array"] = atom_array
    feat["token_array"] = token_array
    feat["masked_prot_restype"] = torch.tensor(
        [masked_residues[x] for i, x in enumerate(feat["restype_id"]) if cond(x, i)]
    )

    feat["prot_residue_mask"] = np.array(
        [1 if token.value in PRO_STD_RESIDUES_VALS_SET else 0 for token in token_array]
    )

    t2 = time.time()

    data = {}
    data["input_feature_dict"] = feat

    # Add dimension related items
    N_token = feat["token_index"].shape[0]
    N_atom = feat["atom_to_token_idx"].shape[0]
    N_msa = feat["msa"].shape[0]

    stats = {}
    for mol_type in ["ligand", "protein", "dna", "rna"]:
        mol_type_mask = feat[f"is_{mol_type}"].bool()
        stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
        stats[f"{mol_type}/token"] = len(torch.unique(feat["atom_to_token_idx"][mol_type_mask]))

    N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
    data.update(
        {
            "N_asym": torch.tensor([N_asym]),
            "N_token": torch.tensor([N_token]),
            "N_atom": torch.tensor([N_atom]),
            "N_msa": torch.tensor([N_msa]),
        }
    )

    def formatted_key(key):
        type_, unit = key.split("/")
        if type_ == "protein":
            type_ = "prot"
        elif type_ == "ligand":
            type_ = "lig"
        else:
            pass
        return f"N_{type_}_{unit}"

    data.update(
        {
            formatted_key(k): torch.tensor([stats[k]])
            for k in [
                "protein/atom",
                "ligand/atom",
                "dna/atom",
                "rna/atom",
                "protein/token",
                "ligand/token",
                "dna/token",
                "rna/token",
            ]
        }
    )
    data.update({"entity_poly_type": entity_poly_type})
    t3 = time.time()
    time_tracker = {
        "crop": t1 - t0,
        "featurizer": t2 - t1,
        "added_feature": t3 - t2,
    }

    return data, atom_array, time_tracker


class InferenceDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
        dump_dir: str,
        task_manager: TaskManager,
        use_msa: bool = True,
        crystal_only: bool = False, 
        seed: Optional[int] = None,
    ) -> None:
        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        self.use_msa = use_msa
        self.task_manager = task_manager
        self.crystal_only = crystal_only # if we only train on crystals, gets rid of unused tokens
        self.seed = None
        with open(self.input_json_path) as f:
            self.inputs = json.load(f)

    def set_seed(self, seed: int):
        self.seed = seed

    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        """
        Processes a single sample from the input JSON to generate features and statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.
        """
        # general features
        sample2feat = SampleDictToFeatures(single_sample_dict, self.task_manager)
        return *build_inference_features(sample2feat, self.use_msa, self.crystal_only), sample2feat

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], AtomArray, str]:
        try:
            single_sample_dict = self.inputs[index]
            sample_name = single_sample_dict["name"]
            logger.info(f"Featurizing {sample_name}...")

            data, atom_array, _, sample2feat = self.process_one(
                single_sample_dict=single_sample_dict
            )

            error_message = ""
        except Exception as e:
            data, atom_array, sample2feat = {}, None, None
            error_message = f"{e}:\n{traceback.format_exc()}"
        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        data["seed"] = self.seed
        return data, atom_array, sample2feat, error_message
