"""
Phase 1.3: Sample Global Sentinel-2 Domain

Use Earth Engine Python API to sample global Sentinel-2 reflectances.
- Filter by year (e.g., 2023)
- Apply Cloud Score+ masking
- Exclude inland waters and ecoregion==0
- Sample from Fibonacci distribution

Output: Global S2 spectral dataset (N_global × 10 bands)
"""

from __future__ import annotations

from pathlib import Path

import ee
import pandas as pd

from gee_pipeline.utilsCloudfree import apply_cloudScorePlus_mask

# Band mapping for Sentinel-2
BAND_NAMES = [
    "B2",  # 10m - Blue (490 nm)
    "B3",  # 10m - Green (560 nm)
    "B4",  # 10m - Red (665 nm)
    "B5",  # 20m - Vegetation Red Edge (705 nm)
    "B6",  # 20m - Vegetation Red Edge (740 nm)
    "B7",  # 20m - Vegetation Red Edge (783 nm)
    "B8",  # 10m - NIR (842 nm)
    "B8A",  # 20m - Vegetation Red Edge (865 nm)
    "B11",  # 20m - SWIR (1610 nm)
    "B12",  # 20m - SWIR (2190 nm)
]


def get_output_dirs(base_dir: Path | None = None) -> tuple[Path, Path]:
    """Returns (data_dir, figures_dir). If base_dir is None, uses script directory."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figures_dir = base_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, figures_dir


def get_s2_indices_for_year(year: int, data_dir: str) -> pd.DataFrame:
    """Reads S2 indices from data/gee_pipeline/outputs/s2_indices_per_mgrs_tile/*.txt for the specified year."""
    # For simplicity, this function assumes a single file with all indices for the year
    # In production, you may want to read multiple files and concatenate
    import glob

    pattern = f"{data_dir}/*_{year}_*.txt"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No S2 indices files found for year {year} with pattern: {pattern}"
        )

    # Read and concatenate all matching files
    dfs = []
    for file in files:
        # readlines from txt (each line has one string system:index)
        with open(file, "r") as f:
            indices = f.read().splitlines()
        df = pd.DataFrame({"system:index": indices})
        dfs.append(df)

    s2_indices = pd.concat(dfs, ignore_index=True)
    return s2_indices


def fetch_global_s2_domain() -> pd.DataFrame:
    """
    Fetch global Sentinel-2 reflectances using Earth Engine API.

    Returns
    -------
    pd.DataFrame
        Global S2 spectra (N_global × 10 bands)

    Notes
    -----
    Uses Cloud Score+ masking with parameters from config/gee_pipeline.yaml:
    - CLOUD_SCORE_PLUS_BAND: 'cs'
    - CLOUD_SCORE_PLUS_THRESHOLD: 0.65
    Applies apply_cloudScorePlus_mask from gee_pipeline.utilsCloudfree.
    """
    data_dir, figures_dir = get_output_dirs()

    # Load Fibonacci sample points
    fibonacci_collection = (
        "projects/ee-speckerfelix/assets/open-earth/fibonacci-samples"
    )
    samples = ee.FeatureCollection(fibonacci_collection)

    # Load S2 data
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

    # Filter by year'
    year = 2020
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    s2_imgc = s2.filterDate(start_date, end_date).filter(
        "CLOUDY_PIXEL_PERCENTAGE < 70"
    )  # Initial filter to reduce data volume

    # read system:indices from data/gee_pipeline/outputs/s2_indices_per_mgrs_tile/*.txt with regex for year 2020
    s2_indices = get_s2_indices_for_year(
        year=year, data_dir="data/gee_pipeline/outputs/s2_indices_per_mgrs_tile"
    )

    # keep a random sample for each MGRS tile: 20200719T031539_20200719T031541_T56XML
    s2_indices["mgrs_tile"] = s2_indices["system:index"].str.split("_").str[2]
    s2_indices_sampled = s2_indices.groupby("mgrs_tile").sample(
        n=1, random_state=42
    )  # Sample single index per MGRS tile for global representativeness

    s2_imgc = s2_imgc.filter(
        ee.Filter.inList("system:index", s2_indices_sampled["system:index"].tolist())
    )

    # Apply Cloud Score+ masking (matching gee_pipeline.yaml configuration)
    cs_band = "cs"  # from CLOUD_FILTERING.CLOUD_SCORE_PLUS_BAND in gee_pipeline.yaml
    cs_threshold = (
        0.65  # from CLOUD_FILTERING.CLOUD_SCORE_PLUS_THRESHOLD in gee_pipeline.yaml
    )
    s2_imgc = apply_cloudScorePlus_mask(
        s2_imgc, cs_band=cs_band, cs_threshold=cs_threshold
    )

    # Create mosaic
    s2_mosaic = s2_imgc.mosaic()

    # now we also reproduce masking:
    water_mask_2020 = ee.ImageCollection("ESA/WorldCover/v200").first()
    s2_masked = s2_mosaic.updateMask(water_mask_2020.neq(80))

    # mask out all rock and ice pixels: ECO_ID = 0 ("Rock and Ice")
    ecoregions = ee.Image(
        "projects/ee-speckerfelix/assets/open-earth/resolve_ecoregions_2017_rasterized"
    ).select("Resolve_Ecoregion")
    s2_masked = s2_masked.updateMask(ecoregions.neq(0))

    # Select bands
    s2_selected = s2_masked.select(BAND_NAMES)

    # Sample with property preservation
    sampled = s2_selected.sampleRegions(
        collection=samples,
        properties=[],
        scale=20,
    )

    # Convert to pandas (may be very large!)
    ee.batch.Export.table.toDrive(
        collection=sampled,
        description=f"global_s2_domain_{year}",
        folder="earth_engine_exports",
        fileNamePrefix=f"global_s2_spectra_{year}",
        fileFormat="CSV",
    ).start()


if __name__ == "__main__":
    ee.Initialize()
    global_s2 = fetch_global_s2_domain()
