"""Declarative settings for the map-to-map comparison notebook."""

from __future__ import annotations

from helpers import ComparisonSpec, Region, SamplingConfig

WORLD_COVER_ASSET = "ESA/WorldCover/v200"
WORLD_COVER_BAND = "Map"
BIOME_ASSET = "projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec"
BIOME_BAND = "Resolve_Biome"
GEE_PROJECT = "ee-speckerfelix"

WORLD_COVER_CLASSES = {
    10: ("Tree cover", "#006400", True),
    20: ("Shrubland", "#ffbb22", True),
    30: ("Grassland", "#ffff4c", True),
    40: ("Cropland", "#f096ff", True),
    50: ("Built-up", "#fa0000", False),
    60: ("Bare / sparse vegetation", "#b4b4b4", True),
    70: ("Snow and ice", "#f0f0f0", False),
    80: ("Permanent water bodies", "#0064c8", False),
    90: ("Herbaceous wetland", "#0096a0", True),
    95: ("Mangroves", "#00cf75", True),
    100: ("Moss and lichen", "#fae6a0", True),
}
INCLUDED_WORLD_COVER_CODES = tuple(
    code for code, (_, _, included) in WORLD_COVER_CLASSES.items() if included
)

COMPARISONS = (
    ComparisonSpec("fapar", "s2biophys", "sl2p", (0.0, 1.0), (-0.35, 0.35)),
    ComparisonSpec("fapar", "s2biophys", "groundedeo", (0.0, 1.0), (-0.35, 0.35)),
    ComparisonSpec("laie", "s2biophys", "sl2p", (0.0, 8.0), (-3.0, 3.0)),
)

# Checkpoint-2 prototype only. Final locations are frozen after the four-site QA step.
PROTOTYPE_REGION = Region(
    name="amazon_brazil_prototype",
    display_name="Amazon tropical forest (prototype)",
    lon=-60.0,
    lat=-3.0,
    window_m=5_000,
    time_start="2021-06-15",
    time_end="2021-08-15",
    ecosystem="tropical evergreen rainforest",
    notes="Candidate region for validating the end-to-end workflow; not yet a final site.",
)

SAMPLING = SamplingConfig(
    year=2021,
    window_m=2_000,
    regions_per_biome=20,
    seed=20210821,
    oversample_factor=5,
    minimum_separation_m=50_000,
    minimum_valid_fraction=0.70,
    batch_size=10,
)

SCALE_M = 20
MAX_CLOUD_COVER = 50
CLOUD_SCORE_PLUS_THRESHOLD = 0.70
