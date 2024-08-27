from typing import List, Optional

import ee
import numpy as np
from geemap import ml
from sklearn.ensemble import RandomForestRegressor


class eeRandomForestRegressor:
    def __init__(
        self,
        feature_names: list[str],
        trait_name: str,
        ee_rf_model: Optional[ee.Classifier],
        # rf_asset_id: Optional[str] = None,
    ):

        self.ee_model = ee_rf_model
        self.feature_names = feature_names
        self.trait_name = trait_name

    def predict_image(self, image: ee.Image) -> ee.Image:
        pred_image = (
            image.select(self.feature_names)
            .classify(self.ee_model)
            .rename(self.trait_name)
        )
        return pred_image

    def predict(
        self, image: ee.Image, copy_properties: Optional[list[str]] = None
    ) -> ee.Image:
        if copy_properties is not None:
            return self.predict_image(image).copyProperties(
                source=image, properties=copy_properties
            )
        else:
            return self.predict_image(image).rename(self.trait_name)
