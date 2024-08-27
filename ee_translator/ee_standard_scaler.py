from typing import List

import ee
from loguru import logger
from sklearn.preprocessing import StandardScaler


class eeStandardScaler:
    def __init__(self, scaler: StandardScaler, feature_names: List[str] | None = None):
        self.scaler_ = scaler
        self.mean_ = scaler.mean_
        self.scale_ = scaler.scale_
        self.ee_mean_ = ee.Array(self.mean_.tolist())  # dim: should be 1D: (n_bands)
        self.ee_scale_ = ee.Array(self.scale_.tolist())  # dim: should be 1D: (n_bands)
        self.feature_names_ = feature_names
        self.n_features_in_ = scaler.n_features_in_
        self.n_samples_seen_ = scaler.n_samples_seen_

        if self.feature_names_ is None:
            logger.debug(
                f"No feature_names provided; they will be set to the names scaler.feature_names_in_"
            )
            self.feature_names_ = list(scaler.feature_names_in_)
        else:
            if not len(self.feature_names_) == len(self.mean_):
                logger.error(
                    f"Length of feature_names: {len(self.feature_names_)} does not match length of mean: {len(self.mean_)}"
                )
                raise ValueError

        # Create ee.Images for the mean and scale
        self.scaler_mean_image = ee.Image.constant(list(self.mean_)).rename(
            self.feature_names_
        )
        self.scaler_scale_image = ee.Image.constant(list(self.scale_)).rename(
            self.feature_names_
        )

    def transform_image(self, image: ee.Image) -> ee.Image:
        # Select the relevant bands
        image_selected = image.select(self.feature_names_)

        # Subtract the mean image and divide by the scale image
        image_scaled = image_selected.subtract(self.scaler_mean_image).divide(
            self.scaler_scale_image
        )

        # Rename the scaled bands to match the original band names
        image_scaled = image_scaled.rename(self.feature_names_)

        # Add the scaled bands to the original image, replacing the original bands
        image_to_return = image.addBands(image_scaled, overwrite=True)

        return image_to_return

    def inverse_transform(self, image: ee.Image) -> ee.Image:
        # TODO: Implement for multiple columns, so far only one column can be backtansformed.
        raise NotImplementedError
        image = image.multiply(self.ee_scale_.get([0]))
        image = image.add(self.ee_mean_.get([0]))
        try:
            image.getInfo()
        except:
            raise ValueError
        return image

    def inverse_transform_column(self, image: ee.Image, column: str) -> ee.Image:
        if not column in self.feature_names_:
            raise ValueError(
                f"Column {column} not in feature names: {self.feature_names_}"
            )

        # get index of column in feature_names
        column_index = self.feature_names_.index(column)

        image_backtransformed = image.select(column).multiply(
            self.ee_scale_.get([column_index])
        )
        image_backtransformed = image_backtransformed.add(
            self.ee_mean_.get([column_index])
        )

        image_to_return = image.addBands(image_backtransformed, overwrite=True)
        return image_to_return
