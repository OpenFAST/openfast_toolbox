"""
Scale selected OpenFAST input variables using openfast_toolbox.

Workflow:
1) Copy the complete source case folder to a target folder.
2) Read the .fst file from the target folder.
3) Follow linked files via FASTInputDeck where possible.
4) Modify selected variables.
5) Write the modified files back in-place, preserving the copied folder structure.

Tested against the uploaded 5MW_Scaling case layout.

Two inputs : 
1. LAMBDA: target to baseline rotor diameter ratio
2. TargetTowerHt: the target tower height in the scaled case. This is used to compute

Scaling rules: 
- Mass ~ LAMBDA^3
- Mass per unit length ~ LAMBDA^2
- Inertia ~ LAMBDA^5
- Bending stiffness ~ LAMBDA^4
- Axial stiffness ~ LAMBDA^2
- Torsional stiffness ~ LAMBDA^4

"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable

# -------------------------------------------------------------------------
# USER SETTINGS
# -------------------------------------------------------------------------
# If openfast_toolbox is not installed with pip, point this to the local clone.
OPENFAST_TOOLBOX_DIR = Path(r"../../repo/openfast_toolbox")

SOURCE_DIR = Path(r"./5MW_Scaling")
TARGET_DIR = Path(r"./Target_scaled")
FST_NAME = "NREL5MW_Scaling.fst"

# Main similarity factor example. Generally rotor diameter ratio. but you can replace/extend this with your own scaling rules.
LAMBDA = 0.60
TargetTowerHt = 80.0 # Target tower height (m). 

# -------------------------------------------------------------------------
# IMPORT OPENFAST TOOLBOX
# -------------------------------------------------------------------------
if OPENFAST_TOOLBOX_DIR.exists():
    sys.path.insert(0, str(OPENFAST_TOOLBOX_DIR.resolve()))

from openfast_toolbox.io import FASTInputDeck, FASTInputFile  # noqa: E402


def copy_case_folder(src: Path, dst: Path, overwrite: bool = True) -> None:
    """Copy the whole OpenFAST case folder to a new working folder."""
    src = src.resolve()
    dst = dst.resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {src}")

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Target folder already exists: {dst}")
        shutil.rmtree(dst)

    shutil.copytree(src, dst)


def find_case_insensitive_file(parent: Path, linked_name: str) -> Path | None:
    """
    Resolve linked OpenFAST filenames even when letter case differs.

    This is useful because Windows is case-insensitive, but Linux/Python path
    resolution is case-sensitive. Example from your case:
    NREL5MW_scaling_ElastoDyn.dat vs NREL5MW_Scaling_ElastoDyn.dat
    """
    linked_name = linked_name.strip().strip('"').replace("\\", "/")
    candidate = parent / linked_name
    if candidate.exists():
        return candidate

    parts = Path(linked_name).parts
    current = parent
    for part in parts:
        if not current.is_dir():
            return None
        matches = [p for p in current.iterdir() if p.name.lower() == part.lower()]
        if not matches:
            return None
        current = matches[0]
    return current


def repair_fst_file_links(fst_path: Path, keys: Iterable[str] = ("EDFile", "AeroFile", "InflowFile", "ServoFile")) -> None:
    """
    Repair simple top-level .fst linked filenames when only letter case differs.
    This keeps FASTInputDeck usable on case-sensitive systems.
    """
    fst = FASTInputFile(str(fst_path))
    case_dir = fst_path.parent
    changed = False

    for key in keys:
        if key not in fst.keys():
            continue
        value = fst[key]
        if not isinstance(value, str):
            continue

        value_clean = value.strip().strip('"')
        if value_clean.lower() in {"unused", "none", "na", "nan"}:
            continue

        resolved = find_case_insensitive_file(case_dir, value_clean)
        if resolved is not None:
            rel = resolved.relative_to(case_dir).as_posix()
            if rel != value_clean:
                fst[key] = f'"{rel}"'
                changed = True
                print(f"Repaired {key}: {value_clean} -> {rel}")

    if changed:
        fst.write(str(fst_path))


def write_file_if_present(obj, path: Path | None) -> None:
    """Write an OpenFAST Toolbox file object if it was read successfully."""
    if obj is not None and path is not None:
        obj.write(str(path))


def main() -> None:
    # 1) Work only in Target_scaled after this point
    copy_case_folder(SOURCE_DIR, TARGET_DIR, overwrite=True)

    target_fst = TARGET_DIR / FST_NAME
    if not target_fst.exists():
        raise FileNotFoundError(f"Cannot find target .fst file: {target_fst}")

    # 2) Repair case-only filename mismatches before reading the full deck
    repair_fst_file_links(target_fst)

    # 3) Read linked OpenFAST files using openfast_toolbox
    # Keep readlist limited to the files you want to access/update.
    deck = FASTInputDeck(
        str(target_fst),
        readlist=["ED", "EDbld", "EDtwr", "AD", "ADbld", "AF"],
        verbose=False,
    )

    print("Files read by FASTInputDeck:")
    for short_key, path in deck.inputFilesRead.items():
        print(f"  {short_key:6s}: {path}")

    # ------------------------------------------------------------------
    # 4) EXAMPLE SCALING MODIFICATIONS
    #    Replace/extend this block with your own scaling rules.
    # ------------------------------------------------------------------
    ED = deck.fst_vt["ElastoDyn"]
    EDbld = deck.fst_vt["ElastoDynBlade"]
    EDtwr = deck.fst_vt["ElastoDynTower"]
    AD = deck.fst_vt["AeroDyn15"]
    ADbld_list = deck.fst_vt["AeroDynBlade"]

    # Example A: scalar values in ElastoDyn
    if ED is not None:
        ED["TipRad"] = ED["TipRad"] * LAMBDA
        ED["HubRad"] = ED["HubRad"] * LAMBDA
        ED["HubCM"] = ED["HubCM"] * LAMBDA
        ED["UndSling"] = ED["UndSling"] * LAMBDA
        ED["OverHang"] = ED["OverHang"] * LAMBDA
        ED["ShftGagL"] = ED["ShftGagL"] * LAMBDA
        ED["NacCMxn"] = ED["NacCMxn"] * LAMBDA
        ED["NacCMyn"] = ED["NacCMyn"] * LAMBDA
        ED["NacCMzn"] = ED["NacCMzn"] * LAMBDA
        ED["NcIMUxn"] = ED["NcIMUxn"] * LAMBDA
        ED["NcIMUyn"] = ED["NcIMUyn"] * LAMBDA
        ED["NcIMUzn"] = ED["NcIMUzn"] * LAMBDA
        ED["Twr2Shft"] = ED["Twr2Shft"] * LAMBDA
        ED["TowerHt"] = TargetTowerHt-ED["Twr2Shft"]
        ED["TowerBsHt"] = ED["TowerBsHt"] * LAMBDA

        ED["HubMass"] = ED["HubMass"] * LAMBDA**3
        ED["HubIner"] = ED["HubIner"] * LAMBDA**5
        ED["GenIner"] = ED["GenIner"] * LAMBDA**5
        ED["NacMass"] = ED["NacMass"] * LAMBDA**3
        ED["NacYIner"] = ED["NacYIner"] * LAMBDA**5

        # Other variables can be edded here for example:
        # ED["GBRatio"] = 50


    # Example B: distributed structural blade properties in ElastoDyn blade file
    # BldProp columns in this file are:
    # 0 BlFract, 1 StrcTwst, 2 BMassDen, 3 FlpStff, 4 EdgStff
    if EDbld is not None:
        EDbld["BldProp"][:, 2] *= LAMBDA**2  # blade mass per length
        EDbld["BldProp"][:, 3] *= LAMBDA**4  # flapwise EI
        EDbld["BldProp"][:, 4] *= LAMBDA**4  # edgewise EI

    # Example C: tower distributed properties in ElastoDyn tower file
    # TowProp columns are:
    # 0 HtFract, 1 TMassDen, 2 TwFAStif, 3 TwSSStif
    if EDtwr is not None:
        EDtwr["TowProp"][:, 1] *= LAMBDA**2  # tower mass per length
        EDtwr["TowProp"][:, 2] *= LAMBDA**4  # tower fore-aft EI
        EDtwr["TowProp"][:, 3] *= LAMBDA**4  # tower side-side EI

    # Example D: aerodynamic blade geometry in AeroDyn blade file
    # BldAeroNodes columns include:
    # 0 BlSpn, 4 BlTwist, 5 BlChord, ...
    for ADbld in ADbld_list or []:
        ADbld["BldAeroNodes"][:, 0] *= LAMBDA  # spanwise node locations
        ADbld["BldAeroNodes"][:, 5] *= LAMBDA  # chord

    # Example E: AeroDyn scalar update, if desired
    if AD is not None:
        # TowProp columns:
        # 0 TwrElev 1 TwrDiam 2 TwrCd 3 TwrTI 4 TwrCb 5 TwrCp 6 TwrCa
        OrgTowerHt=AD["TowProp"][-1, 0]
        AD["TowProp"][:, 0] *= ED["TowerHt"]/OrgTowerHt   # scale TwrDiam
        AD["TowProp"][:, 1] *= LAMBDA   # scale TwrDiam

    # ------------------------------------------------------------------
    # 5) Write modified files back IN PLACE.
    #    This preserves original file names and relative folder structure.
    # ------------------------------------------------------------------
    files = deck.inputFilesRead

    write_file_if_present(deck.fst_vt["Fst"], Path(files.get("Fst")) if files.get("Fst") else None)
    write_file_if_present(ED, Path(files.get("ED")) if files.get("ED") else None)
    write_file_if_present(EDbld, Path(files.get("EDbld")) if files.get("EDbld") else None)
    write_file_if_present(EDtwr, Path(files.get("EDtwr")) if files.get("EDtwr") else None)
    write_file_if_present(AD, Path(files.get("AD")) if files.get("AD") else None)

    # FASTInputDeck stores AeroDyn blade files as a list. In this NREL 5MW case,
    # all three blade entries point to the same physical file, so write unique paths only.
    ad_bld_paths = []
    if "ADbld" in files:
        ad_bld_paths = [Path(files["ADbld"])]
    for ADbld, path in zip(ADbld_list or [], ad_bld_paths):
        write_file_if_present(ADbld, path)

    print(f"\nDone. Scaled case written under: {TARGET_DIR.resolve()}")


if __name__ == "__main__":
    main()
