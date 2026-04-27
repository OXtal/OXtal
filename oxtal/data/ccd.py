import functools
import logging
import multiprocessing
import pickle
import re
import subprocess
import tempfile

from itertools import islice
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Optional, Union

import biotite
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import gemmi
import numpy as np
import rdkit
from biotite.structure import AtomArray
from pdbeccdutils.core import ccd_reader
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem.Descriptors import NumRadicalElectrons
from tqdm import tqdm

from configs.configs_data import data_configs
from oxtal.data.constants import (
    MASK_ONE_LETTER,
    MASK_RESNAME,
    N_BACKBONE_ATOMS,
    PRO_STD_RESIDUES,
)
from oxtal.data.utils import preprocess_smiles, make_coordinate_bonds
from oxtal.data.substructure_perms import get_substructure_perms

logger = logging.getLogger(__name__)

COMPONENTS_FILE = Path(data_configs.get("ccd_components_file", ""))
RKDIT_MOL_PKL = Path(data_configs.get("ccd_components_rdkit_mol_file", ""))

# CCDC SMILES support ---------------------------------------------------
# A text file where each line is either:
#   <SMILES>                     or
#   <SMILES> <identifier>
# Blank lines and lines starting with '#' are ignored.
CCDC_SMILES_FILE = Path(data_configs.get("ccdc_smiles_file", ""))
CCDC_RKDIT_MOL_PKL = Path(data_configs.get("ccdc_components_rdkit_mol_file", "ccdc_rdkit_mols.pkl"))

USE_XTB = bool(data_configs.get("ccdc_use_xtb", True))
# How many threads to give xTB (defaults to available cores)
XTB_THREADS = int(data_configs.get("xtb_threads", 2))
SCRATCH_PATH = data_configs.get("xtb_scratch_path", None)

_MAX_CPU = max(multiprocessing.cpu_count() - 1, 1)
# -----------------------------------------------------------------------------
#                 === CCD‑specific helpers ===
# -----------------------------------------------------------------------------

@functools.lru_cache
def biotite_load_ccd_cif() -> pdbx.CIFFile:
    """biotite load CCD components file

    Returns:
        pdbx.CIFFile: ccd components file
    """
    return pdbx.CIFFile.read(COMPONENTS_FILE)


@functools.lru_cache
def gemmi_load_ccd_cif() -> gemmi.cif.Document:
    """gemmi load CCD components file

    Returns:
        Document: gemmi ccd components file
    """
    return gemmi.cif.read(COMPONENTS_FILE)


def _map_central_to_leaving_groups(component) -> Optional[dict[str, list[list[str]]]]:
    """map each central atom (bonded atom) index to leaving atom groups in component (atom_array).

    Returns:
        dict[str, list[list[str]]]: central atom name to leaving atom groups (atom names).
    """
    comp = component.copy()
    # Eg: ions
    if comp.bonds is None:
        return {}
    central_to_leaving_groups = defaultdict(list)
    for c_idx in np.flatnonzero(~comp.leaving_atom_flag):
        bonds, _ = comp.bonds.get_bonds(c_idx)
        for l_idx in bonds:
            if comp.leaving_atom_flag[l_idx]:
                comp.bonds.remove_bond(c_idx, l_idx)
                group_idx = struc.find_connected(comp.bonds, l_idx)
                if not np.all(comp.leaving_atom_flag[group_idx]):
                    return None
                central_to_leaving_groups[comp.atom_name[c_idx]].append(
                    comp.atom_name[group_idx].tolist()
                )
    return central_to_leaving_groups


@functools.lru_cache
def get_component_atom_array(
    ccd_code: str, keep_leaving_atoms: bool = False, keep_hydrogens=False
) -> AtomArray:
    """get component atom array

    Args:
        ccd_code (str): ccd code
        keep_leaving_atoms (bool, optional): keep leaving atoms. Defaults to False.
        keep_hydrogens (bool, optional): keep hydrogens. Defaults to False.

    Returns:
        AtomArray: Biotite AtomArray of CCD component
            with additional attribute: leaving_atom_flag (bool)
    """
    # Just return glycine if we're masking. We'll fix the ref features later.
    if ccd_code == MASK_RESNAME:
        ccd_code = "GLY"
    if not COMPONENTS_FILE.exists():
        return None
    ccd_cif = biotite_load_ccd_cif()
    if ccd_code not in ccd_cif:
        logger.warning(f"Warning: get_component_atom_array() can not parse {ccd_code}")
        return None
    try:
        comp = pdbx.get_component(ccd_cif, data_block=ccd_code, use_ideal_coord=True)
    except biotite.InvalidFileError as e:
        # Eg: UNL without atom.
        logger.warning(f"Warning: get_component_atom_array() can not parse {ccd_code} for {e}")
        return None
    atom_category = ccd_cif[ccd_code]["chem_comp_atom"]
    leaving_atom_flag = atom_category["pdbx_leaving_atom_flag"].as_array()
    comp.set_annotation("leaving_atom_flag", leaving_atom_flag == "Y")

    for atom_id in ["alt_atom_id", "pdbx_component_atom_id"]:
        comp.set_annotation(atom_id, atom_category[atom_id].as_array())
    if not keep_leaving_atoms:
        comp = comp[~comp.leaving_atom_flag]
    if not keep_hydrogens:
        # EG: ND4
        comp = comp[~np.isin(comp.element, ["H", "D"])]

    # Map central atom index to leaving group (atom_indices) in component (atom_array).
    comp.central_to_leaving_groups = _map_central_to_leaving_groups(comp)
    if comp.central_to_leaving_groups is None:
        logger.warning(
            f"Warning: ccd {ccd_code} has leaving atom group bond to more than one central atom, central_to_leaving_groups is None."
        )
    return comp


@functools.lru_cache(maxsize=None)
def get_one_letter_code(ccd_code: str) -> Union[str, None]:
    """get one_letter_code from CCD components file.

    normal return is one letter: ALA --> A, DT --> T
    unknown protein: X
    unknown DNA or RNA: N
    other unknown: None
    some ccd_code will return more than one letter:
    eg: XXY --> THG

    Args:
        ccd_code (str): _description_

    Returns:
        str: one letter code
    """
    if ccd_code == MASK_RESNAME:
        return MASK_ONE_LETTER
    if not COMPONENTS_FILE.exists():
        return None
    ccd_cif = biotite_load_ccd_cif()
    if ccd_code not in ccd_cif:
        return None
    one = ccd_cif[ccd_code]["chem_comp"]["one_letter_code"].as_item()
    if one == "?":
        return None
    else:
        return one


@functools.lru_cache(maxsize=None)
def get_mol_type(ccd_code: str) -> str:
    """get mol_type from CCD components file.

    based on _chem_comp.type
    http://mmcif.rcsb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp.type.html

    not use _chem_comp.pdbx_type, because it is not consistent with _chem_comp.type
    e.g. ccd 000 --> _chem_comp.type="NON-POLYMER" _chem_comp.pdbx_type="ATOMP"
    https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v5_next.dic/Items/_struct_asym.pdbx_type.html

    Args:
        ccd_code (str): ccd code

    Returns:
        str: mol_type, one of {"protein", "rna", "dna", "ligand"}
    """
    if ccd_code == MASK_RESNAME:
        ccd_code = "GLY"
    if not COMPONENTS_FILE.exists():
        return "ligand"
    ccd_cif = biotite_load_ccd_cif()
    if ccd_code not in ccd_cif:
        return "ligand"

    link_type = ccd_cif[ccd_code]["chem_comp"]["type"].as_item().upper()

    if "PEPTIDE" in link_type and link_type != "PEPTIDE-LIKE":
        return "protein"
    if "DNA" in link_type:
        return "dna"
    if "RNA" in link_type:
        return "rna"
    return "ligand"


def get_all_ccd_code() -> list:
    """get all ccd code from components file"""
    ccd_cif = biotite_load_ccd_cif()
    return list(ccd_cif.keys())

# -----------------------------------------------------------------------------
#             === CCD RDKit preprocessing ===
# -----------------------------------------------------------------------------

def _get_component_rdkit_mol_processing(ccd_code: str, xtb: bool = False, threads: int = 1, charge: int | None = None) -> Union[Chem.Mol, None]:
    """get rdkit mol by PDBeCCDUtils
    https://github.com/PDBeurope/ccdutils

    Args:
        ccd_code (str): ccd code

    Returns
        rdkit.Chem.Mol: rdkit mol with ref coord
    """
    ccd_cif = gemmi_load_ccd_cif()
    try:
        ccd_block = ccd_cif[ccd_code]
    except KeyError:
        return None
    ccd_reader_result = ccd_reader._parse_pdb_mmcif(ccd_block, sanitize=True)
    mol = ccd_reader_result.component.mol

    # Atom name from ccd, reading by pdbeccdutils
    # Copy atom name for pickle https://github.com/rdkit/rdkit/issues/2470
    mol.atom_map = {atom.GetProp("name"): atom.GetIdx() for atom in mol.GetAtoms()}

    mol.name = ccd_code
    mol.sanitized = ccd_reader_result.sanitized
    # First conf is ideal conf.
    mol.ref_conf_id = 0
    mol.ref_conf_type = "idea"

    num_atom = mol.GetNumAtoms()
    # Eg: UNL without atom
    if num_atom == 0:
        return mol

    # Make ref_mask, ref_mask is True if ideal coord is valid
    atoms = ccd_block.find(
        "_chem_comp_atom.", ["atom_id", "model_Cartn_x", "pdbx_model_Cartn_x_ideal"]
    )
    assert num_atom == len(atoms)
    ref_mask = np.zeros(num_atom, dtype=bool)
    for row in atoms:
        atom_id = gemmi.cif.as_string(row["_chem_comp_atom.atom_id"])
        atom_idx = mol.atom_map[atom_id]
        x_ideal = row["_chem_comp_atom.pdbx_model_Cartn_x_ideal"]
        ref_mask[atom_idx] = x_ideal != "?"
    mol.ref_mask = ref_mask

    mol.mulliken_charge = np.zeros(num_atom, dtype=float)
    if not mol.sanitized:
        return mol
    options = rdkit.Chem.AllChem.ETKDGv3()
    options.clearConfs = False
    options.randomSeed=0xf00d
    options.useRandomCoords=True
    try:
        # FIXME: LOOK INTO CHARGES, TAUTOMERS, ISSUES AND OTHER DATA ISSUES
        # FIXME: WHERE IS THE HYDROGEN REMOVED WITH CCD? NEVER DROPPED?
        conf_id = rdkit.Chem.AllChem.EmbedMolecule(mol, options)
        if conf_id < 0:
            raise ValueError(f"EmbedMolecule failed for {ccd_code}")
        mol.ref_conf_id = conf_id
        mol.ref_conf_type = "rdkit"
        if xtb:
            mol_opt = opt_mol(mol, ccd_code, charge, threads, conf_id=conf_id)
            if mol_opt:
                mol = mol_opt # only if it's not None
        mol.ref_mask[:] = True
    except ValueError:
        # Sanitization issue here
        logger.warning(f"Warning: fail to generate conf for {ccd_code}, use idea conf")
        pass
    return mol


# =============================================================================
#                    3D EMBEDDING / OPTIMISATION HELPERS
# =============================================================================

def _ff_optimize(mol: Chem.Mol, workdir: Path, conf_id: int = 0) -> Path:
    """
    Geometry optimisation with MMFF94‑s when possible,
    otherwise UFF.  Returns the path of the minimised
    structure (‹mmff.sdf› or ‹uff.sdf›).

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule (must already contain a 3‑D conformer).
    workdir : pathlib.Path
        Directory in which the SDF file will be written.
    conf_id : int, optional
        Conformer ID to optimise (default 0).

    Raises
    ------
    ValueError
        If neither MMFF nor UFF covers all atoms in `mol`.
    """
    # ----- choose force field -------------------------------------------------
    if AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        ff     = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        method = "mmff"
        ff.Minimize()
    elif AllChem.UFFHasAllMoleculeParams(mol):
        ff     = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        method = "uff"
        ff.Minimize()
    else:
        method = 'etkdg'
        logger.warning(f"Neither MMFF nor UFF has parameters for this molecule, using default ETKDG conformer")

    # ----- write result -------------------------------------------------------
    sdf_path = workdir / f"{method}.sdf"
    with Chem.SDWriter(str(sdf_path)) as w:
        w.write(mol, confId=conf_id)

    return sdf_path

def _xtb_optimize(
    sdf_in: Path,
    workdir: Path,
    threads: int = 1,
    charge: int = 0,
    xtb_version: str | int = "2",
    unpaired_e: int = 0,
    xtb_path: str = "xtb",
) -> tuple[Path, np.ndarray]:
    """Geometry refinement with GFN-xTB returning (SDF path, Mulliken charges)."""
    gfn_flag = ["--gfnff"] if str(xtb_version).lower() == "gfnff" else ["--gfn", str(xtb_version)]
    unrestricted = ["--uhf", str(unpaired_e)] if unpaired_e > 0 else [] # TODO: ALWAYS ASSUMING HIGH SPIN STATE
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        xtb_path,
        sdf_in,
        "--opt",
        "--parallel",
        str(threads),
        "--namespace",
        "geom",
        "--charge",
        str(charge),
    ] + gfn_flag + unrestricted

    with (workdir / "xtb.log").open("w") as log:
        subprocess.run(cmd, cwd=workdir, check=True, stdout=log, stderr=subprocess.STDOUT)

    sdf_out = workdir / "geom.xtbopt.sdf" 
    if not sdf_out.exists():
        raise RuntimeError("xTB did not produce geom.xtbopt.xyz – see xtb.log")

    # -------------------- parse Mulliken charges ----------------------------
    charges_file = workdir / "geom.gfnff_charges" if str(xtb_version).lower() == "gfnff" else workdir / "geom.charges"
    with open(charges_file, 'r') as file:
        lines = file.readlines()
        mulliken = [0]*len(lines)
        for i, line in enumerate(lines):
            mulliken[i] = float(line.split()[0]) 
    mulliken = np.array(mulliken, dtype=float)

    # -------------------- build new RDKit molecule -------------------------
    return sdf_out, mulliken

def opt_mol(mol, name, charge: int | None = None, threads: int = 1, conf_id: int = 0, scratch_path = SCRATCH_PATH):
    # Assumes we already have an embedded mol

    charge = charge if charge else Chem.GetFormalCharge(mol)
    unpaired_e = NumRadicalElectrons(mol)

    # Setup scratch directory
    if scratch_path:
        scratch_path.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(tempfile.mkdtemp(dir=scratch_path, prefix=f"opt_mol_{name}_"))
        should_cleanup = False
    else:
        tmp_dir = tempfile.TemporaryDirectory(prefix=f"opt_mol_{name}_")
        tmp_path = Path(tmp_dir.name)
        should_cleanup = True

    try:
        # ff_sdf = _ff_optimize(mol, tmp_path, conf_id, threads, charge=charge, unpaired_e=unpaired_e)
        # Note, some organometallics seem to just suck with xtb using etkdg?
        sdf_path = tmp_path / f"etkdg.sdf"
        with Chem.SDWriter(str(sdf_path)) as w:
            w.write(mol, confId=conf_id)
        
        xtb_sdf, mulliken = _xtb_optimize(sdf_path, tmp_path / "gfnff", xtb_version='gfnff', threads=threads, charge=charge, unpaired_e=unpaired_e)
        mol.ref_conf_type = "gfnff"
        try:
            xtb_sdf, mulliken = _xtb_optimize(xtb_sdf, tmp_path / "xtb", xtb_version='2', threads=threads, charge=charge, unpaired_e=unpaired_e)
        except Exception as exc:
            logger.warning("xTB optimisation failed for '%s': %s", name, exc)
        
        mol_opt = Chem.SDMolSupplier(str(xtb_sdf), removeHs=False)[0]
        if mol_opt is None:
            logger.warning("Could not read xTB‑optimised SDF for '%s'", name)
            return None
        
        mol_opt = _transfer_attrs(mol, mol_opt) 
        mol = mol_opt
        # Attach Mulliken charges
        if len(mulliken) == mol.GetNumAtoms():
            for idx, at in enumerate(mol.GetAtoms()):
                at.SetDoubleProp("mulliken_charge", float(mulliken[idx]))
            mol.mulliken_charge = mulliken
        else:
            raise ValueError(
                f"Mulliken vector length ({len(mulliken)}) does not match atom count ({mol_opt.GetNumAtoms()}) for '{name}'"
            )

        mol.ref_conf_type = "xtb"
        return mol

    finally:
        # Clean up temporary directory if created
        if not scratch_path and should_cleanup and tmp_dir in locals():
            tmp_dir.cleanup()

# -----------------------------------------------------------------------------
#                 === CCDC SMILES preprocessing ===
# -----------------------------------------------------------------------------

def _transfer_attrs(src: Chem.Mol,
                    dst: Chem.Mol,
                    attrs=("atom_map", "ref_mask", "ref_conf_id", "ref_conf_type", "sanitized", "mulliken_charge", "name")) -> Chem.Mol:
    """
    Copy selected ad-hoc attributes from *src* to *dst* and return *dst*.

    RDKit clones keep C++ properties (SetProp/SetDoubleProp/…),
    but any Python attributes you added disappear.  This helper
    keeps the few we rely on elsewhere.
    """
    for attr in attrs:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
    return dst

def _is_smiles(identifier: str) -> bool:
    """Heuristically decide whether *identifier* is a SMILES string."""

    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(identifier, sanitize=True)
    RDLogger.EnableLog('rdApp.*')
    if mol is not None:
        return True          # definitive yes

    # RDKit could not parse ⇒ probably not SMILES.
    # Most CCD ids are exactly three upper-case letters or digits.
    return not (len(identifier) == 3 and identifier.isupper() and identifier.isalnum())

def _read_ccdc_smiles_file() -> list[tuple[str, str]]:
    """Return list of (smiles, identifier) from *CCDC_SMILES_FILE*."""
    if not CCDC_SMILES_FILE or not CCDC_SMILES_FILE.exists():
        logger.warning("CCDC SMILES file '%s' not configured/existing.", CCDC_SMILES_FILE)
        return []
    smiles_entries: list[tuple[str, str]] = []
    with CCDC_SMILES_FILE.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line, maxsplit=1)
            smiles = parts[0]
            identifier = parts[1] if len(parts) == 2 else smiles # _canonical_smiles(smiles) # FIXME: assume canonicalized 
            smiles_entries.append((smiles, identifier))
    return smiles_entries


def _embed_smiles_to_mol(smiles: str, name: str, keep_hydrogen: bool = False, use_xtb: bool = False, threads: int = 1, charge: int | None = None) -> Union[Chem.Mol, None]:
    mol = Chem.MolFromSmiles(smiles, sanitize = False) # FIXME: assume canonicalized 
    mol = make_coordinate_bonds(mol)
    err = Chem.SanitizeMol(mol, catchErrors=True)
    if err != Chem.SANITIZE_NONE or not mol:
        logger.warning(f"Sanitize failed ({err}) for {name}")
        return None
    num_atom = mol.GetNumAtoms() # this is pre adding/removing Hs
    if num_atom == 0:
        return mol

    # canonicalize atom order, necessary even with canonical SMILES
    # this is ok before AddHs and removeHs as Hs are added at the end
    ranks = [(rank, atom_idx) for atom_idx, rank in enumerate(Chem.CanonicalRankAtoms(mol))]
    ranks.sort()
    mol = Chem.RenumberAtoms(mol,[atom_idx for _,atom_idx in ranks]) # This gets rid of all properties

    mol = Chem.AddHs(mol)

    ps = AllChem.ETKDGv3()
    ps.randomSeed=0xf00d
    ps.useRandomCoords=True
    try:
        status = AllChem.EmbedMolecule(mol, ps)
    except Exception as e:
        logger.warning(f"mol {name} cannot generate a conformer")
        return None
    if status != 0:
        # configure fallback parameters
        # FIXME maybe those we just use xTB FF?
        ps.useBasicKnowledge = False
        status = AllChem.EmbedMolecule(mol, ps)
        if status != 0:
            logger.warning(f"mol {name} cannot generate a conformer")
            return None

    mol.ref_conf_type = "rdkit"
    if use_xtb:
        # Runs xTB and gets the mulliken charges
        mol_opt = opt_mol(mol, name, charge, threads, conf_id=0)
        if mol_opt:
            mol = mol_opt # only if it's not None, this also updates ref_conf_type

    mol.name = name
    if not keep_hydrogen:
        # TODO: hook this up to parser.py, ok for now since we always remove H?
        mol_no_h = Chem.RemoveHs(mol, sanitize=False, updateExplicitCount=True)
        mol_no_h = _transfer_attrs(mol, mol_no_h) 
        mol = mol_no_h

    # 4) Rename *each* atom as: SYMBOL + “occurrence count” (in this new canonical order)
    sym_counter: Counter[str] = Counter()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol().upper()
        sym_counter[sym] += 1
        atom.SetProp("name", f"{sym}{sym_counter[sym]}") # Previously: a.SetProp("name", f"{a.GetSymbol()}{i}")
    # 5) Build the atom_map from name → index
    mol.atom_map = {atom.GetProp("name"): atom.GetIdx() for atom in mol.GetAtoms()}

    mol.sanitized = True
    mol.ref_conf_id = 0
    mol.ref_mask = np.ones(mol.GetNumAtoms(), dtype=bool)
    return mol


def _process_smiles_chunk(smiles_chunk: list[tuple[str, str]], output_path: Path, keep_hydrogen: bool, use_xtb: bool, xtb_threads: int):
    """
    Processes a chunk of SMILES and saves the resulting dictionary to a pickle file.
    It does NOT return the dictionary to avoid IPC overhead.
    """
    # Disable logging within the worker process to keep stdout clean.
    logger.setLevel(logging.ERROR)
    RDLogger.DisableLog('rdApp.*')
    
    results = {}
    for smiles, name in smiles_chunk:
        mol = _embed_smiles_to_mol(smiles, name, keep_hydrogen, use_xtb, xtb_threads, charge=None)
        if mol:
            results[name] = mol
            
    # Save results directly to a file instead of returning them
    with open(output_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def _chunk_iterator(iterable, chunk_size: int):
    """Yields successive chunks of a given size from an iterable."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            return
        yield chunk


def _preprocess_all_ccdc_smiles(keep_hydrogen: bool = False, chunk_size: int = 1000, use_threads: bool = True) -> dict[str, Chem.Mol]:
    """Precompute RDKit molecules for every SMILES in *CCDC_SMILES_FILE* using a chunking strategy."""
    logger.info("Pre‑processing CCDC SMILES list from '%s'", CCDC_SMILES_FILE)
    entries = _read_ccdc_smiles_file()

    num_entries = len(entries)
    logger.info(f"Loaded {num_entries} RDKit molecules")

    chunks = list(_chunk_iterator(entries, chunk_size))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        futures = []
        with ProcessPoolExecutor(max_workers=_MAX_CPU, max_tasks_per_child=1) as executor: #
            # Submit all jobs to the pool
            for i, chunk in enumerate(chunks):
                output_path = temp_path / f"chunk_{i}.pkl"
                future = executor.submit(
                    _process_smiles_chunk,
                    chunk,
                    output_path,
                    keep_hydrogen=keep_hydrogen,
                    use_xtb=USE_XTB,
                    xtb_threads=XTB_THREADS,
                )
                futures.append(future)

            # Use tqdm to show progress as jobs are completed
            logger.info(f"Processing {num_entries} molecules in {len(futures)} chunks...")
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing molecule chunks"):
                try:
                    future.result()  # Check for exceptions from the worker
                except Exception as e:
                    logger.error(f"A worker process failed: {e}")

        # Aggregate results from temporary files after all workers are done
        logger.info("Aggregating results from chunks...")
        all_mols = {}
        for i in range(len(chunks)):
            chunk_file = temp_path / f"chunk_{i}.pkl"
            if chunk_file.exists():
                with open(chunk_file, "rb") as f:
                    mol_dict = pickle.load(f)
                    all_mols.update(mol_dict)

    logger.info(f"Finished processing. Saving {len(all_mols)} molecules to pickle file.")
    with open(CCDC_RKDIT_MOL_PKL, "wb") as fh:
        pickle.dump(all_mols, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return all_mols


# -----------------------------------------------------------------------------
#                  === RDKit molecule caches ===
# -----------------------------------------------------------------------------

_ccd_rdkit_mols: dict[str, Chem.Mol] = {}
_ccdc_rdkit_mols: dict[str, Chem.Mol] = {}


# -----------------------------------------------------------------------------
#                  === Unified RDKit mol accessor ===
# -----------------------------------------------------------------------------


def get_component_rdkit_mol(identifier: str) -> Union[Chem.Mol, None]:
    """Return an RDKit molecule corresponding to *identifier*.

    *identifier* may be:
      • A 3‑letter CCD code (e.g. 'ATP')
      • A raw or canonical SMILES string (processed via CCDC list or on‑the‑fly)
    """
    global _ccd_rdkit_mols, _ccdc_rdkit_mols

    # ----- 0. Fast path: already cached --------------------------------------
    if identifier in _ccd_rdkit_mols:
        return _ccd_rdkit_mols[identifier]
    if identifier in _ccdc_rdkit_mols:
        return _ccdc_rdkit_mols[identifier]

    # Decide CCD vs SMILES -----------------------------------------------------
    if not _is_smiles(identifier): 
        # -------------------- CCD branch --------------------
        # Lazy load entire CCD cache on first miss
        if not _ccd_rdkit_mols:
            if RKDIT_MOL_PKL.exists():
                with RKDIT_MOL_PKL.open("rb") as fh:
                    _ccd_rdkit_mols = pickle.load(fh)
            else:
                # Build cache
                logger.info(f"Generating RDKit mols for all CCD components in {COMPONENTS_FILE} – first run.")
                logger.info("pre-load cif file before multiprocessing avoid read file at each process.")
                gemmi_load_ccd_cif()  # warm cache
                mols = {}
                codes = get_all_ccd_code()
                # FIXME: let's fix this later.
                with multiprocessing.Pool(_MAX_CPU) as pool:
                    for mol in tqdm.tqdm(
                        pool.starmap(
                            _get_component_rdkit_mol_processing, 
                            [(code, USE_XTB, XTB_THREADS, None) for code in codes]),
                        total=len(codes),
                        smoothing=0,
                    ):
                        if mol is not None:
                            mols[mol.name] = mol
                _ccd_rdkit_mols = mols
                with RKDIT_MOL_PKL.open("wb") as fh:
                    pickle.dump(mols, fh)
                logger.info("Saved CCD RDKit mol cache to '%s'", RKDIT_MOL_PKL)
        return _ccd_rdkit_mols.get(identifier, None)

    # ------------------------ SMILES branch ------------------------------
    smiles = identifier
    canon = preprocess_smiles(smiles)

    # Load or build CCDC cache lazily on first SMILES request
    if not _ccdc_rdkit_mols:
        if CCDC_RKDIT_MOL_PKL.is_file():
            with CCDC_RKDIT_MOL_PKL.open("rb") as fh:
                _ccdc_rdkit_mols = pickle.load(fh)
        else:
            if CCDC_SMILES_FILE.is_file():
                _ccdc_rdkit_mols = _preprocess_all_ccdc_smiles()
            else:
                _ccdc_rdkit_mols = {}
    # Try to fetch from cache
    if canon in _ccdc_rdkit_mols:
        return _ccdc_rdkit_mols[canon]

    # Fall‑back: on‑the‑fly embedding of arbitrary SMILES ------------------
    mol = _embed_smiles_to_mol(canon, canon, use_xtb=USE_XTB, threads=XTB_THREADS)
    if mol is not None:
        _ccdc_rdkit_mols[canon] = mol
    return mol

# -----------------------------------------------------------------------------
#               === Reference info extraction (extended) ===
# -----------------------------------------------------------------------------

@functools.lru_cache
def _private_get_ccd_ref_info(identifier: str, *, return_perm: bool = True) -> dict[str, Any]:
    """
    Ref: AlphaFold3 SI Chapter 2.8
    Reference features. Features derived from a residue, nucleotide or ligand’s reference conformer.
    Given an input CCD code or SMILES string, the conformer is typically generated
    with RDKit v.2023_03_3 [25] using ETKDGv3 [26]. On error, we fall back to using the CCD ideal coordinates,
    or finally the representative coordinates
    if they are from before our training date cut-off (2021-09-30 unless otherwise stated).
    At the end, any atom coordinates still missing are set to zeros.

    Get reference atom mapping and coordinates.

    Args:
        name (str): CCD name
        return_perm (bool): return atom permutations.
        # smiles (bool): whether ccd is actually a SMILES
        # do_xtb (bool): use GFN2-xTB or g-xTB to compute the reference geometry & mulliken charges

    Returns:
        Dict:
            ccd: ccd code
            atom_map: atom name to atom index
            coord: atom coordinates
            charge: atom formal charge
            perm: atom permutation
    """
    mol = get_component_rdkit_mol(identifier)
    if mol is None:
        return {}
    if mol.GetNumAtoms() == 0:  # eg: "UNL"
        logger.warning(
            f"Warning: mol {identifier} from get_component_rdkit_mol() has no atoms,"
            "get_ccd_ref_info() return empty dict"
        )
        return {}
    conf = mol.GetConformer(mol.ref_conf_id)
    coord = conf.GetPositions()
    charge = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])

    if not hasattr(mol, "mulliken_charge"):
        mol.mulliken_charge = np.zeros(mol.GetNumAtoms(), dtype=float)

    results: dict[str, Any] = {
        "ccd": identifier,
        "atom_map": mol.atom_map,  # dict[str,int]: atom name to atom index
        "coord": coord,  # np.ndarray[float]: atom coordinates, shape:(n_atom,3)
        "mask": mol.ref_mask,  # np.ndarray[bool]: atom mask, shape:(n_atom,)
        "charge": charge,  # np.ndarray[int]: atom formal charge, shape:(n_atom,)
        "mulliken_charge": mol.mulliken_charge, # np.ndarray[int]: atom Mulliken charge, shape:(n_atom,)
    }

    if return_perm:
        try:
            Chem.SanitizeMol(mol)
            perm = get_substructure_perms(mol, MaxMatches=1000)

        except:
            # Sanitize failed, permutation is unavailable
            perm = np.array(
                [[i for i, atom in enumerate(mol.GetAtoms()) if atom.GetAtomicNum() != 1]]
            )
        # np.ndarray[int]: atom permutation, shape:(n_atom_wo_h, n_perm)
        results["perm"] = perm.T

    return results


# Modified from biotite to use consistent ccd components file
def _connect_inter_residue(atoms: AtomArray, residue_starts: np.ndarray) -> struc.BondList:
    """
    Create a :class:`BondList` containing the bonds between adjacent
    amino acid or nucleotide residues.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The structure to create the :class:`BondList` for.
    residue_starts : ndarray, dtype=int
        Return value of
        ``get_residue_starts(atoms, add_exclusive_stop=True)``.

    Returns
    -------
    BondList
        A bond list containing all inter residue bonds.
    """

    bonds = []

    atom_names = atoms.atom_name
    res_names = atoms.res_name
    res_ids = atoms.res_id
    chain_ids = atoms.chain_id

    # Iterate over all starts excluding:
    #   - the last residue and
    #   - exclusive end index of 'atoms'
    for i in range(len(residue_starts) - 2):
        curr_start_i = residue_starts[i]
        next_start_i = residue_starts[i + 1]
        after_next_start_i = residue_starts[i + 2]

        # Check if the current and next residue is in the same chain
        if chain_ids[next_start_i] != chain_ids[curr_start_i]:
            continue
        # Check if the current and next residue
        # have consecutive residue IDs
        # (Same residue ID is also possible if insertion code is used)
        if res_ids[next_start_i] - res_ids[curr_start_i] > 1:
            continue

        # Get link type for this residue from RCSB components.cif
        curr_link = get_mol_type(res_names[curr_start_i])
        next_link = get_mol_type(res_names[next_start_i])

        if curr_link == "protein" and next_link in "protein":
            curr_connect_atom_name = "C"
            next_connect_atom_name = "N"
        elif curr_link in ["dna", "rna"] and next_link in ["dna", "rna"]:
            curr_connect_atom_name = "O3'"
            next_connect_atom_name = "P"
        else:
            # Create no bond if the connection types of consecutive
            # residues are not compatible
            continue

        # Index in atom array for atom name in current residue
        # Addition of 'curr_start_i' is necessary, as only a slice of
        # 'atom_names' is taken, beginning at 'curr_start_i'
        curr_connect_indices = np.where(
            atom_names[curr_start_i:next_start_i] == curr_connect_atom_name
        )[0]
        curr_connect_indices += curr_start_i

        # Index in atom array for atom name in next residue
        next_connect_indices = np.where(
            atom_names[next_start_i:after_next_start_i] == next_connect_atom_name
        )[0]
        next_connect_indices += next_start_i

        if len(curr_connect_indices) == 0 or len(next_connect_indices) == 0:
            # The connector atoms are not found in the adjacent residues
            # -> skip this bond
            continue

        bonds.append((curr_connect_indices[0], next_connect_indices[0], struc.BondType.SINGLE))

    return struc.BondList(atoms.array_length(), np.array(bonds, dtype=np.uint32))


def build_bb_only_mol(ref_info):
    mol = Chem.RWMol()

    rev_atom_map = {v: k for k, v in ref_info["atom_map"].items()}
    for i in range(N_BACKBONE_ATOMS):
        atom = Chem.Atom(rev_atom_map[i][0])
        idx = mol.AddAtom(atom)

    mol.AddBond(0, 1, Chem.rdchem.BondType.SINGLE)
    mol.AddBond(1, 2, Chem.rdchem.BondType.SINGLE)
    mol.AddBond(2, 3, Chem.rdchem.BondType.DOUBLE)

    conf = Chem.Conformer(N_BACKBONE_ATOMS)
    for i, coord in enumerate(ref_info["coord"][:N_BACKBONE_ATOMS]):
        conf.SetAtomPosition(i, Point3D(*coord))

    mol.AddConformer(conf)
    return mol.GetMol()


def get_coord(mol):
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    coords = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]

    return coords


def get_mask_ref_pos():
    ccd_ref_infos = {ccd: _private_get_ccd_ref_info(ccd) for ccd in PRO_STD_RESIDUES.keys()}
    del ccd_ref_infos["UNK"]

    mols = [build_bb_only_mol(ccd_ref_info) for ccd_ref_info in ccd_ref_infos.values()]

    ref_mol = mols[0]
    for mol in mols[1:]:
        Chem.rdMolAlign.AlignMol(mol, ref_mol)

    all_bb_ref_coords = np.stack([get_coord(mol) for mol in mols])
    mean_bb_coords = all_bb_ref_coords.mean(axis=0)

    return (
        {"N": 0, "CA": 1, "C": 2, "O": 3},
        mean_bb_coords,
        ccd_ref_infos["GLY"]["charge"][:N_BACKBONE_ATOMS],
        ccd_ref_infos["GLY"]["mask"][:N_BACKBONE_ATOMS],
    )

_mask_ref_cache: dict = {}


def _get_mask_ref() -> dict:
    if not _mask_ref_cache:
        atom_map, pos, charge, mask = get_mask_ref_pos()
        _mask_ref_cache.update({"atom_map": atom_map, "pos": pos, "charge": charge, "mask": mask})
    return _mask_ref_cache


def get_ccd_ref_info(identifier: str, *, return_perm: bool = True) -> dict[str, Any]:
    """Return reference information for a CCD code **or** SMILES string.

    The function now transparently accepts either format. For the mask residue
    placeholder the original behaviour is preserved.
    """
    if identifier == MASK_RESNAME:
        ref = _get_mask_ref()
        return {
            "ccd": MASK_RESNAME,
            "atom_map": ref["atom_map"],
            "coord": ref["pos"],
            "charge": ref["charge"],
            "mask": ref["mask"],
        }
    return _private_get_ccd_ref_info(identifier, return_perm=return_perm)

def add_inter_residue_bonds(
    atom_array: AtomArray,
    exclude_struct_conn_pairs: bool = False,
    remove_far_inter_chain_pairs: bool = False,
) -> AtomArray:
    """
    add polymer bonds (C-N or O3'-P) between adjacent residues based on auth_seq_id.

    exclude_struct_conn_pairs: if True, do not add bond between adjacent residues already has non-standard polymer bonds
                  on atom C or N or O3' or P.

    remove_far_inter_chain_pairs: if True, remove inter chain (based on label_asym_id) bonds that are far away from each other.

    returns:
        AtomArray: Biotite AtomArray merged inter residue bonds into atom_array.bonds
    """
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    inter_bonds = _connect_inter_residue(atom_array, res_starts)

    if atom_array.bonds is None:
        atom_array.bonds = inter_bonds
        return atom_array

    select_mask = np.ones(len(inter_bonds._bonds), dtype=bool)
    if exclude_struct_conn_pairs:
        for b_idx, (atom_i, atom_j, b_type) in enumerate(inter_bonds._bonds):
            atom_k = atom_i if atom_array.atom_name[atom_i] in ("N", "O3'") else atom_j
            bonds, types = atom_array.bonds.get_bonds(atom_k)
            if len(bonds) == 0:
                continue
            for b in bonds:
                if (
                    # adjacent residues
                    abs((res_starts <= b).sum() - (res_starts <= atom_k).sum()) == 1
                    and atom_array.chain_id[b] == atom_array.chain_id[atom_k]
                    and atom_array.atom_name[b] not in ("C", "P")
                ):
                    select_mask[b_idx] = False
                    break

    if remove_far_inter_chain_pairs:
        if not hasattr(atom_array, "label_asym_id"):
            logging.warning(
                "label_asym_id not found, far inter chain bonds will not be removed"
            )
        for b_idx, (atom_i, atom_j, b_type) in enumerate(inter_bonds._bonds):
            if atom_array.label_asym_id[atom_i] != atom_array.label_asym_id[atom_j]:
                coord_i = atom_array.coord[atom_i]
                coord_j = atom_array.coord[atom_j]
                if np.linalg.norm(coord_i - coord_j) > 2.5:
                    select_mask[b_idx] = False

    # filter out removed_inter_bonds from atom_array.bonds
    remove_bonds = inter_bonds._bonds[~select_mask]
    remove_mask = np.isin(atom_array.bonds._bonds[:, 0], remove_bonds[:, 0]) & np.isin(
        atom_array.bonds._bonds[:, 1], remove_bonds[:, 1]
    )
    atom_array.bonds._bonds = atom_array.bonds._bonds[~remove_mask]

    # merged normal inter_bonds into atom_array.bonds
    inter_bonds._bonds = inter_bonds._bonds[select_mask]
    atom_array.bonds = atom_array.bonds.merge(inter_bonds)
    return atom_array


def res_names_to_sequence(res_names: list[str]) -> str:
    """convert res_names to sequences {chain_id: canonical_sequence} based on CCD

    Return
        str: canonical_sequence
    """
    seq = ""
    for res_name in res_names:
        one = get_one_letter_code(res_name)
        one = "X" if one is None else one
        one = "X" if len(one) > 1 else one
        seq += one
    return seq