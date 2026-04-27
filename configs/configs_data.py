# pylint: disable=C0114,C0301
import os

DATA_ROOT_DIR = os.environ.get("DATA_ROOT_DIR", "")

CCD_COMPONENTS_FILE_PATH = os.path.join(DATA_ROOT_DIR, "components.cif")
CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
    DATA_ROOT_DIR, "components.cif.rdkit_mol.pkl"
)

data_configs = {
    "ccd_components_file": CCD_COMPONENTS_FILE_PATH,
    "ccd_components_rdkit_mol_file": CCD_COMPONENTS_RDKIT_MOL_FILE_PATH,
}
