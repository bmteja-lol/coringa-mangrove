import ee
from configs.constants import (
    GEE_PROJECT, CORINGA_BBOX, S2_START, S2_END, L8_START, L8_END,
    S2_BANDS, L8_BANDS, MANGROVE_CLASS
)


def init_ee():
    """Initialize Earth Engine. Run once at start."""
    ee.Initialize(project=GEE_PROJECT)


def get_region():
    """Return Coringa bounding box as ee.Geometry."""
    return ee.Geometry.Rectangle(CORINGA_BBOX)


def load_sentinel2():
    """Load and composite Sentinel-2 SR Harmonized for the study period.

    Source: COPERNICUS/S2_SR_HARMONIZED
    Resolution: 10m (visible/NIR), 20m (SWIR)
    Filter: cloud cover < 20%, median composite
    """
    region = get_region()
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region)
          .filterDate(S2_START, S2_END)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .median()
          .clip(region))
    return s2


def load_landsat8():
    """Load and composite Landsat-8 OLI Collection 2 Level 2.

    Source: LANDSAT/LC08/C02/T1_L2
    Resolution: 30m
    Scaling: multiply by 0.0000275, offset -0.2 (Collection 2光学缩放)
    """
    region = get_region()
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(region)
          .filterDate(L8_START, L8_END)
          .filter(ee.Filter.lt("CLOUD_COVER", 20))
          .median()
          .clip(region))
    # Apply Collection 2 Level 2 scaling factors
    scaled = l8.multiply(0.0000275).add(-0.2)
    return scaled


def load_worldcover():
    """Load ESA WorldCover 2021 and create binary mangrove mask.

    Source: ESA/WorldCover/v200/2021
    Mangrove class: 95
    """
    region = get_region()
    wc = (ee.ImageCollection("ESA/WorldCover/v200/2021")
          .filterBounds(region)
          .first()
          .clip(region))
    mangrove_mask = wc.eq(MANGROVE_CLASS).rename("mangrove")
    return mangrove_mask


def get_features_image(sensor="s2"):
    """Load sensor image and compute spectral indices.

    Args:
        sensor: 's2' for Sentinel-2, 'l8' for Landsat-8

    Returns:
        ee.Image with all feature bands
    """
    from src.data.indices import compute_indices_s2, compute_indices_l8

    if sensor == "s2":
        raw = load_sentinel2().select(S2_BANDS)
        return compute_indices_s2(raw)
    elif sensor == "l8":
        raw = load_landsat8().select(L8_BANDS)
        return compute_indices_l8(raw)
    else:
        raise ValueError(f"Unknown sensor: {sensor}")
