import time
from typing import Any, Optional

import torch
import torch.nn as nn

from openfold.model.primitives import LayerNorm

from oxtal.data.json_to_feature import SampleDictToFeatures
from oxtal.model import sample_confidence
from oxtal.model.generator import (
    InferenceNoiseScheduler,
    sample_diffusion,
)
from oxtal.model.utils import simple_merge_dict_list

from oxtal.utils.logger import get_logger
from oxtal.utils.permutation.permutation import SymmetricPermutation
from oxtal.utils.torch_utils import autocasting_disable_decorator

from .modules.diffusion import DiffusionModule
from .modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
from .modules.head import DistogramHead
from .modules.pairformer import PairformerStack, TemplateEmbedder
from .modules.primitives import LinearNoBias

logger = get_logger(__name__)


class oxtalV1Architecture(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs

        # Some constants
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed

        # Diffusion scheduler
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        self.input_embedder = InputFeatureEmbedder(**configs.model.input_embedder)
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
            
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)

        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(in_features=self.c_s, out_features=self.c_z)
        self.linear_no_bias_zinit2 = LinearNoBias(in_features=self.c_s, out_features=self.c_z)
        self.linear_no_bias_token_bond = LinearNoBias(in_features=1, out_features=self.c_z)
        self.linear_no_bias_z_cycle = LinearNoBias(in_features=self.c_z, out_features=self.c_z)
        self.linear_no_bias_s = LinearNoBias(in_features=self.c_s, out_features=self.c_s)
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        task_info,
        xt_noised_struct: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        # device = next(self.pairformer_stack.parameters()).device
        input_feature = {
            "ref_pos": 3,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
            "token_bonds": 1,
        }
        for key, __ in input_feature.items():
            input_feature_dict[key] = input_feature_dict[key].to("cuda")

        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            # Deepspeed_evo_attention do not support token <= 16
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        # Line 1-5
        # try:
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 451]
        input_feature_dict["token_bonds"] = input_feature_dict["token_bonds"].to("cuda")
        s_init = self.linear_no_bias_sinit(s_inputs)  # [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  # [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict)
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.relative_position_encoding(input_feature_dict)
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)
        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training and cycle_no == (N_cycle - 1)
            ):
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))

                if inplace_safe:
                    if self.template_embedder.n_blocks is not None:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                            use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                            and deepspeed_evo_attention_condition_satisfy,
                            use_lma=self.configs.use_lma,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                else:
                    if self.template_embedder.n_blocks is not None:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                            use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                            and deepspeed_evo_attention_condition_satisfy,
                            use_lma=self.configs.use_lma,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                s = (
                    s_init
                    + self.linear_no_bias_s(self.layernorm_s(s))
                    + self.linear_no_bias_sinit(s_inputs)
                )  # ... assuming this is ok... @NOTE: confirm this

                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                    and deepspeed_evo_attention_condition_satisfy,
                    use_lma=self.configs.use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        return s_inputs, s, z


    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )

        if "mode" in kwargs:
            diffusion_method = sample_diffusion
            del kwargs["mode"]

        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            diffusion_method
        )(**_configs, **kwargs)


    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
        sample2feat: SampleDictToFeatures = None,
        task_dict: dict[str, Any] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles of trunk.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        pred_dicts = []
        log_dicts = []
        time_trackers = []
        for _ in range(N_model_seed):
            pred_dict, log_dict, time_tracker = self._main_inference_inner_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                N_sample=self.configs.sample_diffusion["N_sample"],
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                sample2feat=sample2feat,
                task_dict=task_dict,
            )
            pred_dicts.append(pred_dict)
            log_dicts.append(log_dict)
            time_trackers.append(time_tracker)

        # Combine outputs of multiple models
        def _cat(dict_list, key):
            if key not in dict_list[0]:
                return None

            out = [x[key] for x in dict_list]
            if key != "atom_array":
                out = torch.cat(out, dim=0)

            return out

        def _list_join(dict_list, key):
            if key not in dict_list[0]:
                return None

            return sum([x[key] for x in dict_list], [])

        all_pred_dict = {
            "coordinate": _cat(pred_dicts, "coordinate"),
            "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
            "full_data": _list_join(pred_dicts, "full_data"),
            "plddt": _cat(pred_dicts, "plddt"),
            "pae": _cat(pred_dicts, "pae"),
            "pde": _cat(pred_dicts, "pde"),
            "decoder_prediction": _cat(
                pred_dicts, "decoder_prediction"
            ),  # DR: added during inference
            "resolved": _cat(pred_dicts, "resolved"),
            "atom_array": _cat(pred_dicts, "atom_array"),
        }

        all_pred_dict = {key: val for key, val in all_pred_dict.items() if val is not None}

        all_log_dict = simple_merge_dict_list(log_dicts)
        all_time_dict = simple_merge_dict_list(time_trackers)
        return all_pred_dict, all_log_dict, all_time_dict

    def _main_inference_inner_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
        N_sample: int = 1,
        sample2feat: SampleDictToFeatures = None,
        task_dict: dict[str, Any] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]

        """
        Main inference loop (single model seed) for the Alphafold3 model.
        """

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            task_info=task_dict,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        if mode == "inference":
            keys_to_delete = []
            for key in input_feature_dict.keys():
                if "template_" in key or key in [
                    "msa",
                    "has_deletion",
                    "deletion_value",
                    "profile",
                    "deletion_mean",
                    "token_bonds",
                ]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del input_feature_dict[key]
            torch.cuda.empty_cache()

        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        # N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )

        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
            mode=mode,
        )

        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        if mode == "inference" and N_token > 2000:
            torch.cuda.empty_cache()

        # Distogram logits: log contact_probs only, to reduce the dimension
        pred_dict["contact_probs"] = sample_confidence.compute_contact_prob(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )  # [N_token, N_token]

        step_confidence = None

        # Permutation: when label is given, permute coordinates and other heads
        if label_dict is not None and symmetric_permutation is not None:
            pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                input_feature_dict=input_feature_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                permute_by_pocket=("pocket_mask" in label_dict)
                and ("interested_ligand_mask" in label_dict),
            )
            last_step_seconds = step_confidence or step_diffusion
            time_tracker.update({"permutation": time.time() - last_step_seconds})


        return pred_dict, log_dict, time_tracker


    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        task_dict: dict[str, Any] = None,
        mode: str = "inference",
        current_step: Optional[int] = None,
        sample2feat: SampleDictToFeatures = None,
        symmetric_permutation: SymmetricPermutation = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        if label_dict is not None:
            assert label_dict["coordinate"].size() == label_full_dict["coordinate"].size()
            label_dict.update(label_full_dict)

        pred_dict, log_dict, time_tracker = self.main_inference_loop(
            input_feature_dict=input_feature_dict,
            label_dict=label_dict,
            N_cycle=self.N_cycle,
            mode=mode,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            N_model_seed=self.N_model_seed,
            symmetric_permutation=symmetric_permutation,
            task_dict=task_dict,
            sample2feat=sample2feat,
        )
        log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
