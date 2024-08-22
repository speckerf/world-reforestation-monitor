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
        rf_model: Optional[RandomForestRegressor] = None,
        rf_asset_id: Optional[str] = None,
    ):
        assert rf_model is not None or rf_asset_id is not None
        if rf_model is not None:
            self.rf_string = ml.rf_to_strings(
                rf_model,
                feature_names=feature_names,
                output_mode="regression",
                processes=1,
            )
            self.ee_model = ml.strings_to_classifier(self.rf_string)
        else:
            feature_collection = ee.FeatureCollection(rf_asset_id)
            self.ee_model = ml.fc_to_classifier(feature_collection)

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
