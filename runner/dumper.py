import os
from pathlib import Path
from typing import Optional

import torch
from biotite.structure import AtomArray

from oxtal.data.utils import save_structure_cif, save_structure_pdb


class DataDumper:
    def __init__(self, base_dir, need_atom_confidence: bool = False):
        self.base_dir = base_dir
        self.need_atom_confidence = need_atom_confidence

    def dump(
        self,
        dataset_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        step=None,
        file_format: str = "cif",
        dump_dir: Optional[str] = None,
        append_preds_dir: bool = True,
    ):
        """
        Dump the predictions and related data to the specified directory.

        Args:
            dataset_name (str): The name of the dataset.
            pdb_id (str): The PDB ID of the sample.
            seed (int): The seed used for randomization.
            pred_dict (dict): The dictionary containing the predictions.
            atom_array (AtomArray): The AtomArray object containing the structure data.
            entity_poly_type (dict[str, str]): The entity poly type information.
            step (int, optional): The step number. Defaults to None.
        """
        if dump_dir is None:
            dump_dir = self._get_dump_dir(dataset_name, pdb_id, seed, step)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

        structure_paths = self.dump_predictions(
            pred_dict=pred_dict,
            dump_dir=dump_dir,
            pdb_id=pdb_id,
            atom_array=atom_array,
            seed=seed,
            entity_poly_type=entity_poly_type,
            file_format=file_format,
            append_preds_dir=append_preds_dir,
        )
        return structure_paths

    def _get_dump_dir(self, dataset_name: str, sample_name: str, seed: int, step: int) -> str:
        """
        Generate the directory path for dumping data based on the dataset name, sample name, and seed.
        """
        if step is not None:
            return os.path.join(
                self.base_dir, dataset_name, f"step_{step}", sample_name, f"seed_{seed}"
            )
        dump_dir = os.path.join(self.base_dir, dataset_name, sample_name, f"seed_{seed}")
        return dump_dir

    def dump_predictions(
        self,
        pred_dict: dict,
        dump_dir: str,
        pdb_id: str,
        atom_array: AtomArray,
        seed: int,
        entity_poly_type: dict[str, str],
        file_format: str = "cif",
        append_preds_dir: bool = True,
    ):
        """
        Dump raw predictions from the model:
            structure: Save the predicted coordinates as CIF files.
            confidence: Save the confidence data as JSON files.
        """
        if append_preds_dir:
            prediction_save_dir = os.path.join(dump_dir, "predictions")
            os.makedirs(prediction_save_dir, exist_ok=True)
        else:
            prediction_save_dir = dump_dir

        # Append seed to filename to avoid overwriting when using multiple seeds
        pdb_id = f"{pdb_id}_seed{seed}"

        # Dump structure
        if "atom_array" in pred_dict:
            atom_array = pred_dict["atom_array"]
            if isinstance(atom_array, list) and len(atom_array) == 1:
                atom_array = atom_array[0]

        structure_paths = self._save_structure(
            pred_coordinates=pred_dict["coordinate"],
            prediction_save_dir=prediction_save_dir,
            sample_name=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            file_format=file_format,
        )

        return structure_paths

    def _save_structure(
        self,
        pred_coordinates: torch.Tensor,
        prediction_save_dir: str,
        sample_name: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        file_format: str = "cif",
        pred_dict=None,
        idx_offset: int = 0,
    ):
        assert atom_array is not None
        N_sample = pred_coordinates.shape[0]
        output_fpaths = []
        for idx in range(N_sample):
            output_fpath = os.path.join(
                prediction_save_dir, f"{sample_name}_sample_{idx+idx_offset}.{file_format}"
            )
            save_fn = None
            if file_format == "cif":
                save_fn = save_structure_cif
            elif file_format == "pdb":
                save_fn = save_structure_pdb
            if pred_dict is not None and "atom_array" in pred_dict:
                save_fn(
                    atom_array=pred_dict["atom_array"][idx],
                    pred_coordinate=pred_coordinates[
                        idx, : len(pred_dict["atom_array"][idx]), :
                    ],  # remove padding
                    output_fpath=output_fpath,
                    entity_poly_type=entity_poly_type,
                    pdb_id=sample_name,
                )
            else:
                save_fn(
                    atom_array=atom_array,
                    pred_coordinate=pred_coordinates[idx, : len(atom_array), :],  # remove padding
                    output_fpath=output_fpath,
                    entity_poly_type=entity_poly_type,
                    pdb_id=sample_name,
                )
            output_fpaths.append(output_fpath)
        return output_fpaths

