import logging
import os
import traceback
from collections import OrderedDict
from contextlib import nullcontext
from os.path import exists as opexists
from os.path import join as opjoin
from pathlib import Path
from typing import Any, Mapping
import time

import hydra
import rootutils
import torch
from huggingface_hub import hf_hub_download
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig

from oxtal.data.infer_data_pipeline import get_inference_dataloader
from oxtal.data.json_to_feature import SampleDictToFeatures
from oxtal.model.oxtal_architecture import oxtalV1Architecture

# eval model
from oxtal.utils.seed import seed_everything
from oxtal.utils.torch_utils import to_device
from runner.dumper import DataDumper
from runner.utils import print_config_tree

rootutils.setup_root(__file__, indicator=".project-root")

logger = logging.getLogger(__name__)


class InferenceRunner:
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(need_atom_confidence=configs.need_atom_confidence)

    def init_env(self) -> None:
        """Init pytorch/cuda envs."""
        self.fabric = Fabric(
            strategy=DDPStrategy(find_unused_parameters=False),
            num_nodes=self.configs.fabric.num_nodes,
            loggers=[hydra.utils.instantiate(logger) for _, logger in self.configs.logger.items()],
        )
        self.print(
            f"Fabric: {self.fabric}, rank: {self.fabric.global_rank}, world_size: {self.fabric.world_size}"
        )
        self.fabric.launch()
        self.device = self.fabric.device
        torch.cuda.set_device(self.device)
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0,8.9"
        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            if env is not None:
                logging.info(
                    "The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time."
                )
        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", None)
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "The kernels will be compiled when fast_layernorm is called for the first time."
            )

        logging.info("Finished init ENV.")

    def init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:        
        self.model = oxtalV1Architecture(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        checkpoint_path = hf_hub_download(
            repo_id="OXtal-CSP/OXtal",
            filename="OXtal.pt",
        )
        self.print(f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}")
        checkpoint = torch.load(checkpoint_path, self.device)

        sample_key = [k for k in checkpoint["model"].keys()][0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {k[len("module.") :]: v for k, v in checkpoint["model"].items()}

        # Backwards compatibility for the decoder_head_pair ->
        new_keys_vals, keys_to_del = [], []
        for key, val in checkpoint["model"].items():
            if "head_pair" in key:
                print("IN IF STATEMETEHETHETH HERE")
                keys_to_del.append(key)
                new_key = key.replace("head_pair", "head_seq_struct")
                new_keys_vals.append((new_key, val))

        for new_key_val, del_key in zip(new_keys_vals, keys_to_del):
            new_key, val = new_key_val

            del checkpoint["model"][del_key]
            checkpoint["model"][new_key] = val

        current = self.model.state_dict()
        filtered = OrderedDict()

        for k, v in checkpoint["model"].items():
            if k in current and v.shape == current[k].shape:
                filtered[k] = v  # → OK: same name & same shape
            else:
                print(
                    f"Skipping '{k}': not found or shape changed "
                    f"(saved {tuple(v.shape)} → current {tuple(current.get(k, torch.empty(0)).shape)})"
                )
        self.model.load_state_dict(
            state_dict=filtered,
            strict=self.configs.load_strict,
        )
        self.model.eval()
        self.print("Finish loading checkpoint.")

    def init_dumper(self, need_atom_confidence: bool = False):
        self.dumper = DataDumper(base_dir=self.dump_dir, need_atom_confidence=need_atom_confidence)

    # Adapted from runner.train.Trainer.evaluate
    @torch.no_grad()
    def predict(
        self, data: Mapping[str, Mapping[str, Any]], sample2feat: SampleDictToFeatures
    ) -> dict[str, torch.Tensor]:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                sample2feat=sample2feat,
                label_full_dict=None,
                label_dict=None,
                mode=self.configs.inference_mode,
            )

        return prediction

    def print(self, msg: str):
        if self.fabric.is_global_zero:
            logging.info(msg)

    def debug(self, msg: str):
        if self.fabric.is_global_zero:
            logging.debug(msg)


@hydra.main(config_path="../configs", config_name="inference.yaml", version_base=None)
def main(configs: DictConfig):
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    print_config_tree(configs, resolve=True)
    # Runner
    runner = InferenceRunner(configs)

    if isinstance(configs.seeds, int):
        configs.seeds = [configs.seeds]
    num_inference_seeds = configs.get("num_inference_seeds")
    if num_inference_seeds is not None:
        configs.seeds = list(range(num_inference_seeds))

    # Data
    logger.info(f"Loading data from\n{configs.input_json_path}")
    dataloader = get_inference_dataloader(
        runner.fabric,
        configs=configs,
        num_eval_seeds=configs.seeds,
    )

    dump_dir = Path(runner.dump_dir)
    cifs_dir = dump_dir / "cifs"
    cifs_dir.mkdir(parents=True, exist_ok=True)

    num_data, curr_seed, pre_log_dicts = len(dataloader.dataset), None, []
    # inference_times = []
    for batch in dataloader:
        try:
            data, atom_array, sample2feat, data_error_message = batch[0]

            if len(data_error_message) > 0:
                logger.info(data_error_message)
                with open(
                    opjoin(runner.error_dir, f"{data['sample_name']}.txt"),
                    "w",
                ) as f:
                    f.write(data_error_message)
                continue

            if data["seed"] != curr_seed:
                curr_seed = data["seed"]
                logger.info(f"Seed: {curr_seed + 1} / {len(configs.seeds)}")
                seed_everything(seed=curr_seed, deterministic=configs.deterministic)

            # add atom_array in data
            data["input_feature_dict"]["atom_array"] = atom_array

            sample_name = data["sample_name"]
            path = Path(f"{runner.dump_dir}/{sample_name}_sample_{curr_seed}.cif")
            if path.exists():
                print(f"{path} already exists -- skipping")
                continue

            logger.info(
                f"[Rank {runner.fabric.global_rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
                f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                f"N_atom {data['N_atom'].item()}"
            )

            prediction = runner.predict(data, sample2feat)

            file_format = "cif"

            atom_array_pre = prediction.get("atom_array")
            atom_array = atom_array_pre[0] if atom_array_pre is not None else atom_array

            structure_path = None
            structure_path = runner.dumper.dump(
                dataset_name="",
                pdb_id=sample_name,
                seed=curr_seed,
                pred_dict=prediction,
                atom_array=atom_array,
                entity_poly_type=data["entity_poly_type"],
                file_format=file_format,
                dump_dir=cifs_dir,
                append_preds_dir=False,
            )[0]

            cif_id = path.name.split(".")[0]
            this_dict = {"cif_id": cif_id, "model_cif_path": str(structure_path[0])}
            pre_log_dicts.append(this_dict)

            logger.info(
                f"[Rank {runner.fabric.global_rank}] {data['sample_name']} succeeded.\n"
                f"Results saved to {configs.dump_dir}"
            )

        except Exception as e:
            error_message = f"[Rank {runner.fabric.global_rank}]{data['sample_name']} {e}:\n{traceback.format_exc()}"
            logger.info(error_message)
            # Save error info
            if opexists(error_path := opjoin(runner.error_dir, f"{sample_name}.txt")):
                os.remove(error_path)
            with open(error_path, "w") as f:
                f.write(error_message)
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
