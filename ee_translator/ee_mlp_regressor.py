from typing import List, Optional

import ee
import numpy as np
from sklearn.neural_network import MLPRegressor


class eeMLPRegressor:
    """
    A custom class to exectue the .predict of scikit-learn's MLPRegressor on Google Earth Engine (GEE) server side.
    Requires a pretrained MLPRegressor an an input, extracts the weights and biases, import them into GEE runs the forward method on server side.


    Parameters:
    -----------
    model : MLPRegressor
        An instance of the scikit-learn MLPRegressor class that has been fitted to data.

    trait_name : str, optional
        Name of the predicted trait. Default is 'trait'.

    Attributes:
    -----------
    model : MLPRegressor
        The provided MLPRegressor instance.

    trait_name : str
        Name of the predicted trait.

    ee_array_weights : List[ee.Array]
        Weights of the neural network layers, converted to GEE ee.Array objects.

    ee_array_biases : List[ee.Array]
        Biases of the neural network layers, converted to GEE ee.Array objects.

    ee_image_weights : List[ee.Image]
        Weights of the neural network layers, converted to GEE ee.Image objects.

    ee_image_biases : List[ee.Image]
        Biases of the neural network layers, converted to GEE ee.Image objects.

    Methods:
    --------
    _apply_activation_function(x: ee.Image or ee.Array, function: str) -> ee.Image or ee.Array:
        Apply the specified activation function to the input.

    _forward_pass_array(ee_X: ee.Array) -> ee.Array:
        Perform a forward pass on an input array through the neural network layers.

    _forward_pass_image(ee_X: ee.Image) -> ee.Image:
        Perform a forward pass on an input image through the neural network layers.

    predict(image: ee.Image or ee.Array) -> ee.Image:
        Predict an input image or array using the trained neural network.

    Notes:
    ------
    TODO's:
    - Only regression implemented so far. Think about creating a similar eeMLPClassifier class. Both classes could inherit some their function from the base class
    - Check if activation functions are correctly implemented
    - Write test cases

    """

    def __init__(self, model: MLPRegressor, trait_name: str = "trait") -> None:
        self.model = model
        self.trait_name = trait_name
        # if predicting on an array
        self.ee_array_weights = [
            ee.Array(weights.tolist()) for weights in self.model.coefs_
        ]
        self.ee_array_biases = [
            ee.Array(biases.tolist()).reshape([1, -1])
            for biases in self.model.intercepts_
        ]
        # if predictin on an image
        self.ee_image_weights = [ee.Image(array) for array in self.ee_array_weights]
        self.ee_image_biases = [ee.Image(array) for array in self.ee_array_biases]
        # activation function
        # if(self.model.activation != 'tanh'):
        #     raise NotImplementedError
        if self.model.out_activation_ != "identity":
            raise NotImplementedError

    def _apply_activation_function(
        self, x: ee.Image | ee.Array, function: str
    ) -> ee.Image | ee.Array:
        if function == "tanh":
            return x.tanh()
        elif function == "identity":
            return x
        elif function == "softmax":
            raise NotImplementedError
        elif function == "logistic" or function == "sigmoid":
            return x.multiply(-1).exp().add(1).pow(-1)  # 1 / (1 + exp(-x))
        elif function.lower() == "relu":
            return x.gt(0).multiply(x)  # max(0, x)
        else:
            raise ValueError

    def _forward_pass_array(self, ee_X: ee.Array) -> ee.Array:
        # this method is supposed to work with a array where the rows correspond to the number of rows, while the column cooresponds to the number of bands
        n_samples, _ = np.array(ee_X.getInfo()).shape
        x = ee_X  # dim: (n_samples, b_bands)
        for i in range(self.model.n_layers_ - 1):
            x = x.matrixMultiply(self.ee_array_weights[i])
            x = x.add(self.ee_array_biases[i].repeat(axis=0, copies=n_samples))
            if i != self.model.n_layers_ - 2:
                x = self._apply_activation_function(x, self.model.activation)
        # apply output activation
        x = self._apply_activation_function(x, self.model.out_activation_)
        return x

    def _forward_pass_image(self, ee_X: ee.Image) -> ee.Image:
        # # convert input image to arrayImage where each pixel holds an array of shape (1, n_bands)
        # array_X_1D = ee_X.toArray() # dim: (n_bands)
        # array_X_2D = array_X_1D.toArray(1) # dim: (n_bands, 1)
        # array_X_2D = array_X_2D.arrayTranspose() # dim: (1, n_bands)
        x = ee_X.toArray().toArray(1).arrayTranspose()  # dim: (1, n_bands)

        for i in range(self.model.n_layers_ - 1):
            x = x.matrixMultiply(
                self.ee_image_weights[i]
            )  # 1st iteration: dim (1, n_bands) x (n_bands, n_nodes)
            x = x.add(self.ee_image_biases[i])  # (1 x n_nodes)

            if i != self.model.n_layers_ - 2:
                x = self._apply_activation_function(x, self.model.activation)

        # apply output activation
        x = self._apply_activation_function(x, self.model.out_activation_)
        # convert back to image with single band with output trait value
        output_image = x.arrayProject([0]).arrayFlatten([[self.trait_name]])
        return output_image

    def predict(
        self, image: ee.Image | ee.Array, copy_properties: Optional[list[str]] = None
    ) -> ee.Image:
        if isinstance(image, ee.Image):
            # that should work now
            if copy_properties is not None:
                return self._forward_pass_image(image).copyProperties(
                    source=image, properties=copy_properties
                )
            else:
                return self._forward_pass_image(image)
        elif isinstance(image, ee.Array):
            raise NotImplementedError
        else:
            raise TypeError
