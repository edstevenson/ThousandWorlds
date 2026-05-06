from __future__ import annotations

CANONICAL_INPUT_NAMES = ["T_star", "F_star", "radius", "gravity", "P_rot", "P0", "CO2", "CH4"]
CANONICAL_FIELD_VARIABLES = [
    "surface_temperature",
    "temperature",
    "specific_humidity",
    "asr_cloudy",
    "olr_cloudy",
    "cloud_fraction",
    "u",
    "v",
]
SINGLE_LEVEL_FIELDS = {"surface_temperature", "asr_cloudy", "olr_cloudy"}
PUBLIC_NAME_ALIASES = {"asr_cloudy": "asr", "olr_cloudy": "olr"}


def build_field_names(nlev: int) -> list[str]:
    return [
        name
        for var in CANONICAL_FIELD_VARIABLES
        for name in ([var] if var in SINGLE_LEVEL_FIELDS else [f"{var}_{k}" for k in range(nlev)])
    ]


def public_name(name: str) -> str:
    parts = name.rsplit("_", 1)
    base = parts[0] if len(parts) == 2 and parts[1].isdigit() else name
    suffix = f"_{parts[1]}" if base != name else ""
    return f"{PUBLIC_NAME_ALIASES.get(base, base)}{suffix}"


def public_field_names(field_names: list[str]) -> list[str]:
    return [public_name(name) for name in field_names]


FIELDS_ALL_OBS = build_field_names(10)
FIELDS_COMPLETE_OBS_ONLY = build_field_names(9)
SUBSET_TO_FIELDS = {
    "all-obs": FIELDS_ALL_OBS,
    "complete-obs-only": FIELDS_COMPLETE_OBS_ONLY,
}
