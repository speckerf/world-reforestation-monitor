import ee


class MinMaxRangeMasker:
    def __init__(self, min_max_dict: dict, tolerance: float = 0.01):
        """
        A class to handle the detection of out-of-range values for satellite image bands
        using Google Earth Engine. It sets up tolerance levels for minimum and maximum values
        from a dictionary of predefined minimum and maximum values for each band and masks out-of-range
        pixels in an image.

        Attributes:
            min_max_dict (dict): A dictionary containing the minimum and maximum values for each band.
            tolerance (float): A percentage tolerance level for the minimum and maximum values.

        Methods:
            ee_image_min_max_masking(image: ee.Image) -> ee.Image:
                Masks out-of-range pixels based on predefined tolerance ranges for each band.
        """
        self.min_max_dict = min_max_dict
        self.min_vals = {k: v["min"] for k, v in min_max_dict.items()}
        self.max_vals = {k: v["max"] for k, v in min_max_dict.items()}
        self.ranges = {k: v["max"] - v["min"] for k, v in min_max_dict.items()}

        # Tolerance thresholds
        self.min_vals_tolerance = {
            k: v - self.ranges[k] * tolerance for k, v in self.min_vals.items()
        }
        self.max_vals_tolerance = {
            k: v + self.ranges[k] * tolerance for k, v in self.max_vals.items()
        }

        # Convert to Earth Engine objects
        self.ee_min_vals = ee.Dictionary(self.min_vals)
        self.ee_max_vals = ee.Dictionary(self.max_vals)
        self.ee_min_vals_tolerance = ee.Dictionary(self.min_vals_tolerance)
        self.ee_max_vals_tolerance = ee.Dictionary(self.max_vals_tolerance)

        self.ee_min_tolerance_image = ee.Image.constant(
            list(self.min_vals_tolerance.values())
        ).rename(list(self.min_vals_tolerance.keys()))

        self.ee_max_tolerance_image = ee.Image.constant(
            list(self.max_vals_tolerance.values())
        ).rename(list(self.max_vals_tolerance.keys()))

        self.band_names = list(self.min_max_dict.keys())
        self.ee_columns = ee.List(self.band_names)

    def ee_mask(self, image):

        selected_image = image.select(self.band_names)
        masked_image = selected_image.updateMask(
            selected_image.gte(self.ee_min_tolerance_image).And(
                selected_image.lte(self.ee_max_tolerance_image)
            )
        )
        return image.addBands(masked_image, overwrite=True)
