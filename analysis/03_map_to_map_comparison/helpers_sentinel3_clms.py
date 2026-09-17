"""Load Copernicus CLMS Sentinel-3 300 m dekadal products into xarray."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
import rioxarray  # noqa: F401
import xarray as xr
from loguru import logger
from pyproj import Geod
from tqdm import tqdm

STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
DEFAULT_CREDENTIALS = Path(__file__).resolve().parents[2] / "auth" / "auth_s3.txt"
GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class ProductConfig:
    stac_collection: str
    band_name: str
    scale_factor: float
    resolution_m: float = 300.0


PRODUCTS: dict[str, ProductConfig] = {
    "lai": ProductConfig("clms_lai_global_300m_10daily_v2_cog", "LAI", 30.0),
    "fapar": ProductConfig("clms_fapar_global_300m_10daily_v2_cog", "FAPAR", 250.0),
    "fcover": ProductConfig("clms_fcover_global_300m_10daily_v2_cog", "FCOVER", 250.0),
}


def read_credentials(path: Path) -> tuple[str, str]:
    """Read the S3 access key and secret from a two-line text file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Credentials file not found: {path}. Expected access key on line 1 "
            "and secret key on line 2."
        )
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(lines) < 2 or not lines[0] or not lines[1]:
        raise ValueError(f"Credentials file {path} must contain two non-empty lines.")
    return lines[0], lines[1]


def parse_iso8601(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_raster_shape(bbox: list[float], resolution_m: float) -> tuple[int, int]:
    """Estimate the number of native-resolution pixels in a lon/lat bbox."""
    west, south, east, north = bbox
    if west >= east or south >= north:
        raise ValueError("Invalid bbox: expected west < east and south < north.")
    mid_lat = (south + north) / 2.0
    mid_lon = (west + east) / 2.0
    width_m = GEOD.line_length([west, east], [mid_lat, mid_lat])
    height_m = GEOD.line_length([mid_lon, mid_lon], [south, north])
    return (
        max(1, math.ceil(width_m / resolution_m)),
        max(1, math.ceil(height_m / resolution_m)),
    )


def get_product_date(item: dict[str, Any]) -> datetime:
    match = re.search(r"_([0-9]{12})_", item["id"])
    if match is None:
        value = item.get("properties", {}).get("datetime")
        if value:
            return parse_iso8601(value)
        raise ValueError(f"Could not extract product date from item ID: {item['id']}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def search_stac_items(
    *,
    stac_collection: str,
    bbox: list[float],
    time_from: str,
    time_to: str,
    limit: int = 500,
    timeout: int = 300,
    id_like: str | None = "%-RT6_%",
) -> list[dict[str, Any]]:
    """Find all dekads in the half-open interval [time_from, time_to)."""
    payload: dict[str, Any] = {
        "collections": [stac_collection],
        "bbox": bbox,
        "datetime": f"{time_from}/{time_to}",
        "limit": limit,
    }
    if id_like:
        payload["filter-lang"] = "cql2-json"
        payload["filter"] = {"op": "like", "args": [{"property": "id"}, id_like]}

    response = requests.post(f"{STAC_URL}/search", json=payload, timeout=timeout)
    response.raise_for_status()
    features = response.json().get("features", [])
    t0, t1 = parse_iso8601(time_from), parse_iso8601(time_to)
    features = [item for item in features if t0 <= get_product_date(item) < t1]
    return sorted(features, key=get_product_date)


def get_item_asset(item: dict[str, Any], product_name: str) -> str:
    """Resolve the raster href without relying on one fixed STAC asset key."""
    assets = item.get("assets", {})
    data_key = f"{product_name}300_{product_name}"
    preferred = [data_key] if data_key in assets else []
    preferred.extend(
        key
        for key in assets
        if product_name in key.lower() and key not in preferred
    )
    for key in preferred + [key for key in assets if key not in preferred]:
        asset = assets[key]
        # The native s3:// href works with the AWSSession configured below;
        # the alternate download URL requires a different bearer token.
        href = asset.get("href") or asset.get("alternate", {}).get("https", {}).get(
            "href"
        )
        if href:
            return href
    raise ValueError(f"No downloadable asset in STAC item {item.get('id')}.")


def fetch_item_array(
    *,
    access_key: str,
    access_secret: str,
    bbox: list[float],
    product_name: str,
    item: dict[str, Any],
) -> xr.DataArray:
    """Read one bbox window directly from a CLMS cloud-optimised GeoTIFF."""
    from rasterio.session import AWSSession

    session = AWSSession(
        aws_access_key_id=access_key,
        aws_secret_access_key=access_secret,
        region_name="default",
    )
    with (
        rasterio.Env(
            session=session,
            AWS_S3_ENDPOINT="eodata.dataspace.copernicus.eu",
            AWS_VIRTUAL_HOSTING=False,
        ),
        rioxarray.open_rasterio(
            get_item_asset(item, product_name), mask_and_scale=True
        ) as src,
    ):
        da = src.rio.clip_box(*bbox).squeeze("band", drop=True).load()

    product_date = np.datetime64(get_product_date(item).replace(tzinfo=None))
    da = da.rename({"y": "latitude", "x": "longitude"})
    da = da.expand_dims(time=[product_date]).transpose("time", "latitude", "longitude")
    da.name = product_name
    return da


def load_clms_s3_xarray(
    *,
    bbox: list[float],
    time_from: str,
    time_to: str,
    product_name: str = "lai",
    credentials: Path = DEFAULT_CREDENTIALS,
    limit: int = 500,
    timeout: int = 300,
    id_like: str | None = "%-RT6_%",
) -> xr.DataArray | None:
    """Load the full CLMS Sentinel-3 dekadal time series for a bbox/date range."""
    if product_name not in PRODUCTS:
        raise ValueError(
            f"Unknown product {product_name!r}; choose {sorted(PRODUCTS)}."
        )
    bbox = [float(value) for value in bbox]
    compute_raster_shape(bbox, PRODUCTS[product_name].resolution_m)
    access_key, access_secret = read_credentials(credentials)
    items = search_stac_items(
        stac_collection=PRODUCTS[product_name].stac_collection,
        bbox=bbox,
        time_from=time_from,
        time_to=time_to,
        limit=limit,
        timeout=timeout,
        id_like=id_like,
    )
    if not items:
        logger.warning(
            "No CLMS {} items in bbox {} from {} to {}",
            product_name,
            bbox,
            time_from,
            time_to,
        )
        return None

    arrays = [
        fetch_item_array(
            access_key=access_key,
            access_secret=access_secret,
            bbox=bbox,
            product_name=product_name,
            item=item,
        )
        for item in tqdm(
            items, desc=f"Fetching CLMS {product_name} ({len(items)} dekads)"
        )
    ]
    result = xr.concat(arrays, dim="time").sortby("time")
    result.attrs.update(
        product=product_name,
        source="Copernicus CLMS Sentinel-3 300 m dekadal v2",
        stac_collection=PRODUCTS[product_name].stac_collection,
    )
    return result

