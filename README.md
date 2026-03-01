# 🔮 OXtal: Generative Molecular Crystal Structure Prediction

This is the official repository for OXtal, a diffusion-based generative model for molecular crystal structure prediction. For more information, please see our ICLR 2026 paper [here](https://arxiv.org/abs/2512.06987).

This code heavily relies on and builds off of the [Protenix](https://github.com/bytedance/Protenix) code base, and we thank the authors of that work for their efforts.

## ⚙️ Installation and Setup
OXtal was developed with Python 3.11.0, CUDA 12.6, and PyTorch 2.5.0, but you may need to adjust these accordingly to match your own compute resources. To set up the environment, run the following commands from the top-level `OXtal` directory, which should create the `oxtal-env` environment in `.venv/`

```
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install oxtal-env
uv sync
```

After installing the environment through `uv`, you can activate it using:
```
source .venv/bin/activate
```

Next, install cutlass:

```
# First clone the cutlass repo
git clone -b v3.5.1 https://github.com/NVIDIA/cutlass.git

# Then, set the environment variable CUTLASS_PATH to point there
cd cutlass
pwd
export CUTLASS_PATH=<path_from_pwd>
```

After that, ensure that you have [GitHub LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage) installed, since that is what we use to manage the large checkpoint files. Finally, you must set the rest of your local variables in `run_inference.sh`. To enable wandb logging, be sure to specify `<PROJECT_NAME>` and `<ENTITY_NAME>` in `configs/logger/wandb.yaml`.


## 🚀 Inference

This project uses [hydra](https://hydra.cc/) to manage model configuration files, which allows easy command-line overrides and structured configs. You can find all the configuration files in the `configs` folder. 

We also use Github LFS in order to manage our model checkpoint and data files, which can be found in `./checkpoints` and `./data` respectively. In order to run inferenc with a different OXtal checkpoint, update the `load_checkpoint_path` parameter in `run_inference.sh`. In order to evaluate using ema, set both `load_checkpoint_path` AND `load_ema_checkpoint_path` to the desired ema checkpoint file.

To generate samples with OXtal, run the following command:
```
# Run OXtal inference:
bash run_inference.sh
```

To run inference on different sets of molecules, simply update the `input_json_path` parameter in `run_inference.sh`. We have provided all of our evaluation datasets in the `examples` folder. To run inference on all 5 evaluation datasets from the paper together, use the `examples/all_inference.json` file.

You can also run OXtal with your own molecules by adding a new .json file to the `examples` folder. The code supports both SMILES strings as well as input .sdf files (see example below). For co-crystals, all component parts must be specified individually, with the desired ratios provided. Example .json entries are provided below for reference.

```
[
    {
        "sequences": [
            {
                "ligand": {
                    "ligand": "CC1=CC(C#N)=C(S1)Nc2c([N+]([O-])=O)cccc2", # SMILES string
                    "count": 30, # How many copies of the molecule to generate
                    "id_key": "ligand"
                }
            }
        ],
        "modelSeeds": [],
        "assembly_id": "1",
        "name": "QAXMEH"
    },
    {
        "sequences": [
            {
                "ligand": {
                    "ligand": "N#CC(C#N)=C1C=CC(C=C1)=C(C#N)C#N",
                    "count": 30,
                    "id_key": "ligand"
                }
            },
            {
                "ligand": {
                    "ligand": "c1cc2cccc3c4cccc5cccc(c(c1)c23)c45",
                    "count": 30,
                    "id_key": "ligand"
                }
            }
        ],
        "modelSeeds": [],
        "assembly_id": "1",
        "name": "PERTCQ01" # Co-Crystal with 1:1 ratio
    },
    {
        "sequences": [
            {
                "ligand": {
                    "ligand": "FILE_./examples/BIPY.sdf", # Input .sdf instead of SMILES string
                    "count": 30,
                    "id_key": "ligand"
                }
            }
        ],
        "modelSeeds": [],
        "assembly_id": "1",
        "name": "UWEQUL"
    },
  ...
]
```

## 📊 Evaluation
We use the COMPACK software from CCDC to compare OXtal generated crystal packings to experimentally observed structures in our test datasets. You can install the `csd-python-api` into your envioronment using the following command:

```
uv pip install --extra-index-url https://pip.ccdc.cam.ac.uk/ csd-python-api
```

After installing, ensure that you have properly installed the CCDC dataset and activated your license. For reference, setup instructions using the command line on a Linux machine can be found [here](https://support.ccdc.cam.ac.uk/support/solutions/articles/103000306299-custom-installation-of-the-csd-portfolio-software-and-data).

After running inference and setting up CCDC, you can simply run:

```
bash run_eval.sh
```
and the evaluation summary report will be generated in `evaluation/metric_summary.txt`. You can modify paths and file names as necessary in `run_eval.sh` as well.

## 📙 Cite
If you make use of this code or its accompanying [paper](https://arxiv.org/abs/2512.06987), please cite this work as follows:
```
@inproceedings{jin2025oxtal,
  title={OXtal: An All-Atom Diffusion Model for Organic Crystal Structure Prediction},
  author={Jin, Emily and Nica, Andrei Cristian and Galkin, Mikhail and Rector-Brooks, Jarrid and Lee, Kin Long Kelvin and Miret, Santiago and Arnold, Frances H and Bronstein, Michael and Bose, Avishek Joey and Tong, Alexander and Liu, Cheng-Hao},
  booktitle={ICLR},
  year={2026}
}
```

## 📄 License
[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

OXtal was trained on data from [CCDC's Cambridge Structural Database](https://www.ccdc.cam.ac.uk/). Therefore, this work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International][cc-by-nc] (CC BY-NC 4.0) License. For commercial use, please ensure that you have a proper [CCDC License](https://www.ccdc.cam.ac.uk/support-and-resources/licensing-information/). 

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

