"""
crystal_similarity.py

COMPACK packing similarity against ground-truth sets *in CSD*
for predicted CIFs named:  [CSD_ID]_[something].cif

CSV mapping:
    - Column: CSD_ID
    - Value:  one or more CSD refcodes (semicolon-separated) that form a polymorph set.
      Example rows:
          JEKVII;JEKVII01
          QQQCIG05;QQQCIG13;QQQCIG14
    - Each member maps to the full set (so JEKVII -> {JEKVII, JEKVII01}, etc.)

Success rule:
    A prediction is a SUCCESS if ≥ ceil(N * [threshold]) molecules in the packing shell
    (default N=15, threshold = 0.5) match to *any single* member of its ground-truth set.
    The reported RMSD is the best (lowest) RMSD among the successful members.

Requires:
    - CSD Python API with CSD-Materials / CSD-Enterprise
    - tqdm

Example:
    python crystal_similarity.py ./preds \
      --truth-map polymorph_sets.csv \
      --packing-size 15 \
      --out-csv packing_summary.csv \
      --distance-tol 0.50 \
      --angle-tol 75.0 \
      --timeout-ms 40000 \
      --workers 128 \
      --save-dir kept_structures \
      -v
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import os
import shutil
import sys
from pathlib import Path
import statistics as stats
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed 
import multiprocessing as mp 
from functools import partial  

from ccdc import io as csdio
from ccdc.io import CrystalReader
from ccdc.crystal import PackingSimilarity


SLUG = "packing_similarity"
DESCRIPTION = "COMPACK packing similarity (success if ≥ half of N matched)"
PRIMARY_METRIC = "rmsd_N"

DEFAULT_PACKING_SIZE = 20         # N in RMSD_N
MATCH_FRACTION = 0.5         # success threshold: ≥ ceil(N * 0.5)

FILENAME_REF_RE = re.compile(r"^(?P<ref>[A-Za-z0-9]+)_.+\.cif$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
#                                   I/O                                       #
# --------------------------------------------------------------------------- #


def _parse_refcode_from_filename(path: Union[str, Path]) -> Optional[str]:
    """Extract CSD refcode from '[CSD_ID]_seedx_sample_[y].cif'."""
    m = FILENAME_REF_RE.match(Path(path).name)
    return m.group("ref") if m else None


def _read_polymorph_map(csv_path: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Read CSV with column 'CSD_ID' containing semicolon-separated CSD refcodes
    that belong to the same polymorph set. Each member maps to the full set.
    """
    mapping: Dict[str, List[str]] = {}
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        if "CSD_ID" not in (r.fieldnames or []):
            raise ValueError("Mapping CSV must contain a 'CSD_ID' column.")
        for row in r:
            raw = (row.get("CSD_ID") or "").strip()
            if not raw:
                continue
            refs = [x.strip() for x in raw.split(";") if x.strip()]
            if not refs:
                continue
            # Map every member -> full set
            for ref in refs:
                mapping[ref.upper()] = [y.upper() for y in refs]
    return mapping


# --------------------------------------------------------------------------- #
#                           Packing Similarity Core                           #
# --------------------------------------------------------------------------- #
 
def _compute_steric_clash_metrics(cryst: "ccdc.crystal.Crystal", *, cutoff: float = 0.2) -> bool:
    """
    Return True if any intermolecular heavy-atom vdW overlap ≥ cutoff exists; else False.
    # FIXME: SEEMS TO RETURN FALSE ON SOME OBVIOUS STUFF?
    """
    mol = cryst.molecule
    overlaps: List[float] = []
    try:
       # contacts() returns vdW-corrected short contacts; negative gap => overlap.
        # We keep intermolecular, heavy-atom contacts only.
        for c in mol.contacts(distance_range=(-5.0, 0.0), only_strongest=False):
            a, b = c.atoms
            if a.atomic_symbol == "H" or b.atomic_symbol == "H":
                continue
            vdw_sum = a.vdw_radius + b.vdw_radius
            d = float(c.length)
            overlap = max(0.0, vdw_sum - d)
            if overlap > 0.0:
                overlaps.append(overlap)
    except Exception:
        # Be robust; no clashes detected if contacts cannot be computed.
        overlaps = []
    
    n_ge_cutoff = sum(1 for x in overlaps if x >= float(cutoff))
    return bool(n_ge_cutoff > 0) 
    # can compute more, e.g.  max_overlap = max(overlaps) if overlaps else 0.0


def run(
    query_file: Union[str, Path],
    truth_refcodes: Iterable[str],
    *,
    packing_size: Union[int, Sequence[int]],
    distance_tol: float,
    angle_tol: float,
    timeout_ms: int,
    allow_mol_diff: bool,
    entry_reader: Optional[csdio.EntryReader] = None,
    clash_cutoff: float,
) -> Dict[str, object]:
    """Compare one query CIF against a set of *CSD* refcodes; return best outcome."""

    p = Path(query_file)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with CrystalReader(str(p)) as rdr:
        if len(rdr) == 0:
            raise ValueError(f"No crystals found in file: {p}")
        q_cryst = rdr[0]
    
    if isinstance(packing_size, int):
        packing_size = [packing_size]
    largest_packing_size = max(packing_size)
    need_matches = math.ceil(largest_packing_size * MATCH_FRACTION)

    entry_reader = entry_reader or csdio.EntryReader()  # default CSD database

    clash = _compute_steric_clash_metrics(q_cryst, cutoff=float(clash_cutoff))
    best = {
        "best_true_refcode": "",
        "results_by_size": {s: {"nmatched": 0, "rmsd": float("inf")} for s in packing_size},
        "passed": False,
        "errors": [],  # list[str]
        "clash": clash,
    }


    for ref in truth_refcodes:

        current_ref_results = {}

        for size in packing_size:
            engine = PackingSimilarity() # unfortunately it needs to be initialized like this... or else the results will override itself
            s = engine.settings
            s.packing_shell_size = size
            if size == 1:
                s.distance_tolerance = 0.2
                s.angle_tolerance = 20
                s.match_entire_packing_shell = True
            else:
                s.distance_tolerance = float(distance_tol)
                s.angle_tolerance = float(angle_tol)
                s.match_entire_packing_shell = False
            s.timeout_ms = int(timeout_ms)
            s.allow_molecular_differences = bool(allow_mol_diff)
            s.ignore_hydrogen_counts = True
            s.ignore_hydrogen_positions = True
            s.ignore_bond_counts = True # this make things pretty slow, but honestly needed
            s.ignore_bond_types = True
            s.allow_artificial_inversion = True


            ref_u = ref.upper()
            # if ref_u == "CUYVUR":
            #     with CrystalReader('/disk2/chenghao/Proteinx-private/protenix/metrics/2339127.cif') as rdr:
            #         t_cryst = rdr[0]
            # else:
            t_cryst = entry_reader.entry(ref_u).crystal

            try:
                comp = engine.compare(t_cryst, q_cryst)
                if comp is None:
                    current_ref_results[size] = {"nmatched": 0, "rmsd": float("inf")}
                else:
                    current_ref_results[size] = {"nmatched": int(comp.nmatched_molecules), "rmsd": float(comp.rmsd)}
            except Exception as e:
                best["errors"].append(f"truth_load_error:{ref_u}:{size}:{e}")
                current_ref_results[size] = {"nmatched": 0, "rmsd": float("inf")}

        res_largest_size = current_ref_results.get(largest_packing_size, {"nmatched": 0, "rmsd": float("inf")})
        nmatched_at_largest = res_largest_size["nmatched"]
        rmsd_at_largest = res_largest_size["rmsd"]

        # Rank: Ranking is based *only* on the largest packing size; successful first (lower RMSD wins), else by higher nmatched
        cand_rank = (
            0 if nmatched_at_largest >= need_matches else 1,
            rmsd_at_largest if nmatched_at_largest >= need_matches else float("inf"),
            -nmatched_at_largest,
        )
        best_res_at_largest = best["results_by_size"][largest_packing_size]
        best_rank = (
            0 if best["passed"] else 1,
            best_res_at_largest["rmsd"] if best["passed"] else float("inf"),
            -best_res_at_largest["nmatched"],
        )

        if cand_rank < best_rank:
            best.update({
                "best_true_refcode": ref_u,
                "results_by_size": current_ref_results,
                "passed": bool(nmatched_at_largest >= need_matches),
            })

    # Set RMSD to NaN if no match was ever found for a given size
    for size in packing_size:
        if best["results_by_size"][size]["rmsd"] == float("inf"):
             best["results_by_size"][size]["rmsd"] = float("nan")
    # print(best)
    # FIXME: best sometimes contain correct things (e.g. ) 'results_by_size': {1: {'nmatched': 1, 'rmsd': 0.22}, 2: {'nmatched': 2, 'rmsd': 0.86}, 15: {'nmatched': 10, 'rmsd': 2.43}}. but other times, it seems to be completely the same for all packing sizes, e.g. 'results_by_size': {1: {'nmatched': 30, 'rmsd': 1.66}, 2: {'nmatched': 30, 'rmsd': 1.66}, 15: {'nmatched': 30, 'rmsd': 1.66}}
    return best


# --------------------------------------------------------------------------- #
#                              Worker function                                #
# --------------------------------------------------------------------------- #

_ENTRY_READER = None  
def _get_entry_reader() -> csdio.EntryReader: 
    """
    Lazily create a single CSD EntryReader per *process*.
    Avoid sharing across processes; it's not picklable and may not be safe.
    """
    global _ENTRY_READER
    if _ENTRY_READER is None:
        # Optional: tame BLAS/OMP oversubscription when you use many workers.
        # These env vars are read on import/first use; set them here as a fallback.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        _ENTRY_READER = csdio.EntryReader()
    return _ENTRY_READER

def _process_file(  
    f: Path,
    truth_set: Sequence[str],
    packing_size: Union[int, Sequence[int]],
    distance_tol: float,
    angle_tol: float,
    timeout_ms: int,
    allow_mol_diff: bool,
    clash_cutoff: float,
) -> Dict[str, object]:
    """
    Run a single file through packing similarity and return the output-row dict.
    Executed in worker processes when args.workers > 1.
    """
    ref = _parse_refcode_from_filename(f)
    if not ref:
        raise ValueError("no reference CSD")

    ref_u = ref.upper()

    # Ensure we use a *per-process* EntryReader
    entry_reader = _get_entry_reader()

    if isinstance(packing_size, int):
        packing_size = [packing_size]
    try:
        result = run(
            f,
            truth_set if truth_set else [ref_u],
            packing_size=packing_size,
            distance_tol=distance_tol,
            angle_tol=angle_tol,
            timeout_ms=timeout_ms,
            allow_mol_diff=allow_mol_diff,
            entry_reader=entry_reader,
            clash_cutoff=clash_cutoff
        )
        row = {
            "file": f.name,
            "csd_refcode": ref_u,
            "best_true_refcode": result["best_true_refcode"],
            "clash": result["clash"],
            "passed": result["passed"],
            "errors": ";".join(result["errors"]) if result["errors"] else "",
        }

        for size, res in result["results_by_size"].items():
            row[f"nmatched_{size}"] = res["nmatched"]
            row[f"rmsd_{size}"] = res["rmsd"]
        return row

    except Exception as e:
        # Harden against any unexpected worker crash; report in CSV.
        row = {
            "file": f.name,
            "csd_refcode": ref_u,
            "best_true_refcode": "",
            "nmatched": 0,
            "packing_shell_size": int(packing_size),
            f"rmsd_{int(packing_size)}": float("nan"),
            "passed": False,
            "errors": f"worker_error:{type(e).__name__}:{e}",
            "clash": None,
        }
        for size in packing_size:
            row[f"nmatched_{size}"] = 0
            row[f"rmsd_{size}"] = float("nan")
        return row


# --------------------------------------------------------------------------- #
#                                   CLI                                       #
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="COMPACK packing similarity vs. CSD ground truths (polymorph sets)."
    )
    ap.add_argument("inputs", nargs="+", help="CIF files or directories")
    ap.add_argument("--truth-map", required=True,
                    help="CSV with column 'CSD_ID' containing semicolon-separated polymorph sets")
    ap.add_argument("--packing-size", type=int, nargs='+', default=[DEFAULT_PACKING_SIZE],
                    help="One or more packing shell sizes N (default: 20)")
    ap.add_argument("--distance-tol", type=float, default=0.50,
                    help="Distance tolerance (fraction, default 0.50)")
    ap.add_argument("--angle-tol", type=float, default=75.0, 
                    help="Angle tolerance in degrees (default 75.0)")
    ap.add_argument("--timeout-ms", type=int, default=100000, # 5000; 100000, ## 1 extra 0 NOTE: LONGER TIME WILL MATCH MORE DISTANTLY RELATED THINGS, KEEP IT AT 5000 FOR QUICK TURNAROUND
                    help="Comparison timeout in ms (0 = none)")
    ap.add_argument("--allow-molecular-differences", action="store_true", # NOTE: DO NOT DO, WILL TAKE FOREVER
                    help="Allow different compounds to be compared (default: off)")
    ap.add_argument("--out-csv", default="packing_summary.csv",
                    help="Write summary CSV here (default: packing_summary.csv)")
    ap.add_argument("--save-dir", default=None,
                    help="If set, move PASSING predictions to this directory")
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of threads for parallel comparisons (default 8)")
    ap.add_argument("--clash-cutoff", type=float, default=0.7,
                    help="Heavy-atom vdW overlap (Å) counted as a steric clash (default 0.70)")  # very short/low-barrier H-bonds (O···O ≈ 2.40 Å) are ≈ 0.64 Å overlap. Halogen bonds and strong charge-assisted contacts rarely exceed ≈ 0.5–0.6 Å
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    # ---------------------------------- Logger ----------------------------------
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("packing_similarity")

    # ---------------------------- Collect input CIFs ----------------------------
    files: List[Path] = []
    for item in args.inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob("*.cif")))
        elif p.suffix.lower() == ".cif":
            files.append(p)
    files = [f for f in files if f.exists()]
    if not files:
        ap.error("No CIF files found among inputs.")

    # Build mapping: member refcode -> full polymorph set
    poly_map = _read_polymorph_map(args.truth_map)
    
    rows: List[Dict[str, object]] = []
    csd_any_pass: Dict[str, bool] = {}

    # Prepare truth sets per file so workers get simple, small payloads.
    truth_per_file: Dict[str, List[str]] = {}
    for f in files:
        ref = _parse_refcode_from_filename(f)
        ref_u = ref.upper()
        truth_per_file[f.name] = poly_map.get(ref_u, [ref_u])

    # ----------------------------- Multiprocessing -----------------------------
    # On platforms that default to 'fork', using 'spawn' can be more robust with C++ libs.
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # already set; fine

    # Prepare a partially-bound worker so the executor only needs (file, truth_set)
    worker = partial(
        _process_file,
        packing_size=args.packing_size,
        distance_tol=args.distance_tol,
        angle_tol=args.angle_tol,
        timeout_ms=args.timeout_ms,
        allow_mol_diff=args.allow_molecular_differences,
        clash_cutoff=args.clash_cutoff,
    )

    future_to_file = {}
    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        for f in files:
            ts = truth_per_file.get(f.name, [_parse_refcode_from_filename(f).upper() if _parse_refcode_from_filename(f) else ""])
            future = ex.submit(worker, f, ts)
            future_to_file[future] = f

        for _ in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Comparing", unit="file"):
            pass  # tqdm progress only

        # Collect results in a deterministic order (file order)
        results_map = {}
        for fut, f in future_to_file.items():
            # try:
            results_map[f.name] = fut.result()
            # except Exception as e:
            #     results_map[f.name] = {
            #         "file": f.name,
            #         "csd_refcode": _parse_refcode_from_filename(f).upper() if _parse_refcode_from_filename(f) else "",
            #         "best_true_refcode": "",
            #         "clash": None,
            #         "passed": False,
            #         "errors": f"result_error:{type(e).__name__}:{e}",
            #     }
            #     for size in args.packing_size:
            #         results_map[f.name][f"nmatched_{size}"] = 0
            #         results_map[f.name][f"rmsd_{size}"] = float("nan")

    # ------------------------------ Get Results ------------------------------
    rows = [results_map[f.name] for f in files]

    # Build pass stats
    for r in rows:
        if r["csd_refcode"]:
            csd_any_pass.setdefault(r["csd_refcode"], False)
            if r["passed"]:
                csd_any_pass[r["csd_refcode"]] = True

    total_csd = len(csd_any_pass)
    csd_pass = sum(1 for v in csd_any_pass.values() if v)
    valid_clash_rows = [r for r in rows if r.get("clash") is not None]
    n_clash = sum(1 for r in valid_clash_rows if r["clash"])
    n_clash_total = len(valid_clash_rows)
    n_pass = sum(1 for r in rows if r["passed"])
    n_total = len(rows)

    # ----------------------------- Write Results -----------------------------
    out_path = Path(args.out_csv)
    with out_path.open("w", newline="") as out:
        fieldnames = [
            "file", "csd_refcode", "best_true_refcode",
            "clash", "passed", "errors",
        ]
        for size in sorted(args.packing_size):
            fieldnames.extend([f"nmatched_{size}", f"rmsd_{size}"])
        w = csv.DictWriter(out, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log.info("Summary written to %s", out_path)

    # ------------------------------ Save passed ------------------------------
    if args.save_dir:
        dst = Path(args.save_dir).expanduser().resolve()
        dst.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if r.get("passed"):
                src = Path(r["file"]).resolve()
                if not src.exists():
                    continue
                link_path = dst / src.name
                if link_path.exists():
                    log.info("Skipping link; already exists: %s", link_path)
                    continue
                # Prefer a symlink to keep originals untouched
                try:   
                    link_path.symlink_to(src)
                except Exception:
                    try:
                        os.link(src, link_path)
                    except Exception:
                        shutil.copy2(src, link_path)
        log.info("Passing structures linked in: %s", dst)

    # ----------------------------- Print stats -----------------------------
    # Print requested pass statistics: by unique CSD_ID (≥1 passing prediction)
    if total_csd:
        log.info("CSD_ID pass rate: %d/%d = %.3f", csd_pass, total_csd, csd_pass / total_csd)
    else:
        log.info("No valid CSD_IDs were processed.")

    if n_total:
        log.info("Prediction pass rate for all structures: %d/%d = %.3f", n_pass, n_total, n_pass / n_total)
        log.info("Clash rate (for %d structures with valid clash checks): %d/%d = %.3f", n_clash_total, n_clash, n_clash_total, n_clash / n_clash_total)

    # Get Best PASS per CSD entry
    largest_packing_size = max(args.packing_size)
    metric_field = f"rmsd_{largest_packing_size}"
    best_by_csd = {}
    for r in rows:
        if not (r.get("passed") and r.get("csd_refcode")):
            continue
        val = r.get(metric_field)
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            continue
        ref = r["csd_refcode"]
        cur = best_by_csd.get(ref)
        if cur is None or val < cur[metric_field]:
            best_by_csd[ref] = r

    if best_by_csd:
        # Print compact summary
        log.info("Best (passing) per CSD_ID:")
        vals = sorted(best_by_csd.values(), key=lambda x: x["csd_refcode"])
        w1 = max(len("CSD_ID"), *(len(d["csd_refcode"]) for d in vals))
        w2 = max(len("file"), *(len(str(d.get("file",""))) for d in vals))
        
        header_parts = [f"{'CSD_ID':<{w1}}", f"{'file':<{w2}}", f"{'clash':<5}"]
        for size in sorted(args.packing_size):
            header_parts.append(f"n/RMSD_{size:<5}")
        log.info("  ".join(header_parts))

        for d in vals:
            row_parts = [f"{d['csd_refcode']:<{w1}}", f"{str(d.get('file','')):<{w2}}", f"{str(d.get('clash')): <5}"]
            for size in sorted(args.packing_size):
                nmatched = d.get(f'nmatched_{size}', '-')
                rmsd = d.get(f'rmsd_{size}', float('nan'))
                cell = f"{nmatched}/{rmsd:.3f}" if not math.isnan(rmsd) else f"{nmatched}/-"
                row_parts.append(f"{cell:<12}")
            log.info("  ".join(row_parts))

        # Save to CSV: <out_csv stem>_best_by_csd.csv
        best_path = out_path.with_name(out_path.stem + "_best_by_csd.csv")
        with best_path.open("w", newline="") as f:
            best_fieldnames = ["csd_refcode", "file", "best_true_refcode", "passed", "clash"]
            for size in sorted(args.packing_size):
                best_fieldnames.extend([f"nmatched_{size}", f"rmsd_{size}"])
            
            w = csv.DictWriter(f, fieldnames=best_fieldnames, extrasaction='ignore')
            w.writeheader()
            w.writerows(vals)
        avg_rmsd = stats.mean(d[metric_field] for d in best_by_csd.values())
        log.info("Average %s across best_by_csd: %.4f", metric_field, avg_rmsd)
        log.info("Best-by-CSD summary written to %s", best_path)


if __name__ == "__main__":
    main()
