"""
Scale selected OpenFAST input variables using openfast_toolbox.

Workflow:
1) Copy the complete source case folder to a target folder.
2) Read the .fst file from the target folder.
3) Follow linked files via FASTInputDeck where possible.
4) Modify selected variables.
5) Write the modified files back in-place, preserving the copied folder structure.

Tested against the uploaded 5MW_Scaling case layout.

Inputs : 


"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable
from dataclasses import dataclass
import math

# -------------------------------------------------------------------------
# USER SETTINGS
# -------------------------------------------------------------------------
# If openfast_toolbox is not installed with pip, point this to the local clone.
OPENFAST_TOOLBOX_DIR = Path(r"../../repo/openfast_toolbox")

SOURCE_DIR = Path(r"./5MW_Scaling")
TARGET_DIR = Path(r"./Target_scaled_clean")
FST_NAME = "NREL5MW_Scaling_Linear.fst"


# -------------------------------------------------------------------------
# SCALING INPUTS AND FACTOR CALCULATION
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class TurbineScalingInput:
    """Minimum turbine data needed by the similarity-scaling method."""

    rated_power_mw: float
    rotor_diameter_m: float
    hub_height_m: float
    rotor_mass_kg: float | None = None
    nacelle_mass_kg: float | None = None
    blade_mass_kg: float | None = None

    @property
    def rotor_radius_m(self) -> float:
        return self.rotor_diameter_m / 2.0

    @property
    def swept_area_m2(self) -> float:
        return math.pi * self.rotor_radius_m**2


@dataclass(frozen=True)
class ScalingFactors:
    """OpenFAST scaling factors derived from the method document."""

    length: float
    specific_thrust_ratio: float

    blade_mass: float
    blade_mass_per_length: float
    blade_stiffness: float
    blade_axial_stiffness: float

    tower_mass: float
    tower_mass_per_length: float
    tower_stiffness: float
    tower_axial_stiffness: float


def blade_scaling_variable(turbine: TurbineScalingInput, specific_thrust_ratio: float = 1.0) -> float:
    """Blade mass scaling variable: r_st * P^(1/3) * R^(3/2)."""
    return specific_thrust_ratio * turbine.rated_power_mw ** (1.0 / 3.0) * turbine.rotor_radius_m ** (3.0 / 2.0)


def tower_scaling_variable(turbine: TurbineScalingInput) -> float:
    """Tower mass scaling variable: P^(1/3) * H^(3/2)."""
    return turbine.rated_power_mw ** (1.0 / 3.0) * turbine.hub_height_m ** (3.0 / 2.0)


def mass_per_length_variable(mass_variable: float, length_m: float) -> float:
    """Convert a mass-like scaling variable to a mass-per-length variable."""
    return mass_variable / length_m


def stiffness_variable(mass_variable: float) -> float:
    """Bending/torsional stiffness scaling variable, proportional to mass_variable^(4/3)."""
    return mass_variable ** (4.0 / 3.0)


def axial_stiffness_variable(mass_variable: float) -> float:
    """Axial stiffness scaling variable, proportional to mass_variable^(2/3)."""
    return mass_variable ** (2.0 / 3.0)


def calculate_specific_thrust_ratio(base: TurbineScalingInput, target: TurbineScalingInput) -> float:
    """Calculate r_st = (P_t^(2/3) / A_t) / (P_b^(2/3) / A_b)."""
    return (
        (target.rated_power_mw ** (2.0 / 3.0) / target.swept_area_m2)
        / (base.rated_power_mw ** (2.0 / 3.0) / base.swept_area_m2)
    )


def calculate_scaling_factors(base: TurbineScalingInput, target: TurbineScalingInput) -> ScalingFactors:
    """Calculate all scaling factors used later in the OpenFAST edits."""
    length = target.rotor_diameter_m / base.rotor_diameter_m
    r_st = calculate_specific_thrust_ratio(base, target)

    base_blade_mass = blade_scaling_variable(base)
    target_blade_mass = blade_scaling_variable(target, specific_thrust_ratio=r_st)

    base_tower_mass = tower_scaling_variable(base)
    target_tower_mass = tower_scaling_variable(target)

    return ScalingFactors(
        length=length,
        specific_thrust_ratio=r_st,
        blade_mass=target_blade_mass / base_blade_mass,
        blade_mass_per_length=mass_per_length_variable(target_blade_mass, target.rotor_radius_m)
        / mass_per_length_variable(base_blade_mass, base.rotor_radius_m),
        blade_stiffness=stiffness_variable(target_blade_mass) / stiffness_variable(base_blade_mass),
        blade_axial_stiffness=axial_stiffness_variable(target_blade_mass)
        / axial_stiffness_variable(base_blade_mass),
        tower_mass=target_tower_mass / base_tower_mass,
        tower_mass_per_length=mass_per_length_variable(target_tower_mass, target.hub_height_m)
        / mass_per_length_variable(base_tower_mass, base.hub_height_m),
        tower_stiffness=stiffness_variable(target_tower_mass) / stiffness_variable(base_tower_mass),
        tower_axial_stiffness=axial_stiffness_variable(target_tower_mass)
        / axial_stiffness_variable(base_tower_mass),
    )


def point_inertia_scale_from_mass(base_mass_kg: float, target_mass_kg: float) -> float:
    """Point inertia scale when actual target mass is available: I ~ m^(5/3)."""
    if base_mass_kg <= 0.0 or target_mass_kg <= 0.0:
        raise ValueError("Base and target masses must be positive for inertia scaling.")
    return (target_mass_kg / base_mass_kg) ** (5.0 / 3.0)


def point_inertia_scale_from_length(length_scale: float) -> float:
    """Fallback point inertia scale when target mass is unavailable: I ~ L^5."""
    return length_scale**5


def target_hub_mass(target_rotor_mass_kg: float, target_blade_mass_kg: float, number_of_blades: int = 3) -> float:
    """Calculate target hub-system mass from rotor mass minus blade masses."""
    hub_mass = target_rotor_mass_kg - number_of_blades * target_blade_mass_kg
    if hub_mass <= 0.0:
        raise ValueError("Calculated target hub mass is not positive. Check rotor and blade masses.")
    return hub_mass


# Base turbine: NREL 5 MW
BASE_TURBINE = TurbineScalingInput(
    rated_power_mw=5.0,
    rotor_diameter_m=126.0,
    hub_height_m=90.0,
)

# Target turbine: REpower MM82 HH100 at La Haute Borne
# Optional inputs : rotor_mass, nacelle_mass, blade_mass
# when target rotor mass is given, 
# i.e. Rotor mass= Hub mass + 3*blade mass 
# target blade mass = S_m * base blade mass.  
TARGET_TURBINE = TurbineScalingInput(
    rated_power_mw=2.0,
    rotor_diameter_m=82.0,
    hub_height_m=100.0,
    rotor_mass_kg=36_000.0,
    nacelle_mass_kg=66_000.0,
    blade_mass_kg=8_695.0,
)

SCALE = calculate_scaling_factors(BASE_TURBINE, TARGET_TURBINE)

# Backward-compatible names used in the OpenFAST edit block below.
LAMBDA = SCALE.length
r_st = SCALE.specific_thrust_ratio
TargetHubHt = TARGET_TURBINE.hub_height_m
TargetRotMass = TARGET_TURBINE.rotor_mass_kg
TargetNacMass = TARGET_TURBINE.nacelle_mass_kg
m_bld_t = TARGET_TURBINE.blade_mass_kg

ScaleBladeMass = SCALE.blade_mass
ScaleBladeMassDen = SCALE.blade_mass_per_length
ScaleBladeStiff = SCALE.blade_stiffness
ScaleBladeStiffAxial = SCALE.blade_axial_stiffness

ScaleTowerMass = SCALE.tower_mass
ScaleTowerMassDen = SCALE.tower_mass_per_length
ScaleTowerStiff = SCALE.tower_stiffness
ScaleTowerStiffAxial = SCALE.tower_axial_stiffness


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
        ED["TowerHt"] = TargetHubHt-ED["Twr2Shft"]
        ED["TowerBsHt"] = ED["TowerBsHt"] * LAMBDA

        # Point masses and inertias. If actual target mass is available, use
        # mass-based inertia scaling. Otherwise use the length-based fallback.
        TargetHubMass = target_hub_mass(TargetRotMass, m_bld_t)
        BaseHubMass = ED["HubMass"]
        BaseNacMass = ED["NacMass"]

        ED["HubMass"] = TargetHubMass
        ED["HubIner"] *= point_inertia_scale_from_mass(BaseHubMass, TargetHubMass)
        ED["GenIner"] *= point_inertia_scale_from_length(LAMBDA)
        ED["NacMass"] = TargetNacMass
        ED["NacYIner"] *= point_inertia_scale_from_mass(BaseNacMass, TargetNacMass)

        # When HubMass, NacMass are not available, use below. 
        # ED["HubMass"] = ED["HubMass"] * LAMBDA**3
        # ED["HubIner"] = ED["HubIner"] * LAMBDA**5
        # ED["GenIner"] = ED["GenIner"] * LAMBDA**5
        # ED["NacMass"] = ED["NacMass"] * LAMBDA**3
        # ED["NacYIner"] = ED["NacYIner"] * LAMBDA**5

        # Other variables can be added here, for example:
        # ED["GBRatio"] = 50


    # Example B: distributed structural blade properties in ElastoDyn blade file
    # BldProp columns in this file are:
    # 0 BlFract, 1 StrcTwst, 2 BMassDen, 3 FlpStff, 4 EdgStff
    if EDbld is not None:
        EDbld["BldProp"][:, 2] *= ScaleBladeMassDen  # blade mass per length
        EDbld["BldProp"][:, 3] *= ScaleBladeStiff  # flapwise EI
        EDbld["BldProp"][:, 4] *= ScaleBladeStiff  # edgewise EI

    # Example C: tower distributed properties in ElastoDyn tower file
    # TowProp columns are:
    # 0 HtFract, 1 TMassDen, 2 TwFAStif, 3 TwSSStif
    if EDtwr is not None:
        EDtwr["TowProp"][:, 1] *= ScaleTowerMassDen  # tower mass per length
        EDtwr["TowProp"][:, 2] *= ScaleTowerStiff  # tower fore-aft EI
        EDtwr["TowProp"][:, 3] *= ScaleTowerStiff  # tower side-side EI

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
        OrgTowerHt = AD["TowProp"][-1, 0]
        AD["TowProp"][:, 0] *= ED["TowerHt"] / OrgTowerHt   # scale TwrElev
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
