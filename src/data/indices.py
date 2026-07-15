import ee
from configs.constants import S2_BANDS, L8_BANDS


def compute_indices_s2(image):
    """Compute 5 spectral indices from Sentinel-2 bands.

    Band mapping: B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2

    Indices:
        NDVI  = (NIR - Red) / (NIR + Red)
        NDMI  = (NIR - SWIR1) / (NIR + SWIR1)
        MNDWI = (Green - SWIR1) / (Green + SWIR1)
        GCVI  = NIR / Green - 1
        SR    = NIR / Red
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    gcvi = image.expression(
        "B8 / B3 - 1",
        {"B8": image.select("B8"), "B3": image.select("B3")}
    ).rename("GCVI")
    sr = image.expression(
        "B8 / B4",
        {"B8": image.select("B8"), "B4": image.select("B4")}
    ).rename("SR")
    return image.addBands([ndvi, ndmi, mndwi, gcvi, sr])


def compute_indices_l8(image):
    """Compute 5 spectral indices from Landsat-8 bands.

    Band mapping: B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1, B7=SWIR2

    Same formulas as S2 but different band letters.
    """
    ndvi = image.normalizedDifference(["B5", "B4"]).rename("NDVI")
    ndmi = image.normalizedDifference(["B5", "B7"]).rename("NDMI")
    mndwi = image.normalizedDifference(["B3", "B6"]).rename("MNDWI")
    gcvi = image.expression(
        "B5 / B3 - 1",
        {"B5": image.select("B5"), "B3": image.select("B3")}
    ).rename("GCVI")
    sr = image.expression(
        "B5 / B4",
        {"B5": image.select("B5"), "B4": image.select("B4")}
    ).rename("SR")
    return image.addBands([ndvi, ndmi, mndwi, gcvi, sr])
