from typing import Callable
from typing import List

import numpy as np
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import streamlit as st
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore

L1 = "L1"
L2 = "L2"
MODEL_PARAM_MAP = {L1: "l1", L2: "l2"}


# TODO: type hints! also, this can't be modelled with Logistic Regression
def true_func(x):
    y = (x**2).sum(1) ** 0.5 < 0.8
    return y.astype(int)


# TODO: replace this class with a function if it only has one method
class DataCreator:
    """
    We want a dataset which is:
    * Easy to visualise: maybe a binary classifier over 2 useful input
    features, along with 2 non-diagnostic features which we don't have to plot
    * Noisy enough that over-fitting is possible (random swaps of the training
    labels)

    Note that over-fitting requires a level of complexity in the model also.
    This might be easiest to achieve with a neural network.
    """

    IV_RANGE = (-1, 1)
    N_SAMPLES = 100

    @classmethod
    def create_data(
        cls,
        n_useful_features: int,
        n_noise_features: int,
        noise_amount: float,
        function: Callable[[np.ndarray], np.ndarray] = true_func,
    ) -> pd.DataFrame:

        if not (0 <= noise_amount <= 1):
            raise ValueError

        useful_features = np.random.uniform(
            *cls.IV_RANGE,
            size=(cls.N_SAMPLES, n_useful_features),
        )
        labels = function(useful_features).reshape(-1, 1)

        # After getting a label based on the "true" value, add error
        useful_features += np.random.uniform(
            -noise_amount,
            noise_amount,
            size=(cls.N_SAMPLES, n_useful_features),
        )
        useful_features = useful_features.clip(*cls.IV_RANGE)

        noise_features = np.random.uniform(
            *cls.IV_RANGE,
            size=(cls.N_SAMPLES, n_noise_features),
        )

        df = pd.DataFrame(
            np.concatenate((useful_features, noise_features), axis=1),
            columns=(
                *(f"x_useful_{i}" for i in range(n_useful_features)),
                *(f"x_noise_{i}" for i in range(n_noise_features)),
            ),
        ).assign(y=labels)

        return df


class Trainer:
    def __init__(
        self,
        model: BaseEstimator,
        data: pd.DataFrame,
        dv: str = "y",
    ) -> None:
        self.X = data[[col for col in data if col != dv]]
        self.y = data[dv]
        self.model = model

    def train(self) -> None:
        self.model.fit(self.X, self.y)

    def predict(self, x: pd.DataFrame) -> List[int]:
        return self.model.predict(x).tolist()


class App:
    @staticmethod
    def main() -> None:
        st.header("L1 vs. L2 Regularisation Intuition")

        # Create data
        data = DataCreator.create_data(2, 2, 0, true_func)

        # Select Regularisation Option
        reg_type = st.selectbox("Choose a regularisation type", (L1, L2))

        # Select Regularisation Strength and other things as appropriate
        st.write("TODO: Select regularisation strength, other parameters?")

        trainer = Trainer(
            LogisticRegression(
                penalty=MODEL_PARAM_MAP[reg_type],
                solver="liblinear",
            ),
            data,
        )

        # Action button
        model_trained = st.button("Fit model")

        if model_trained:
            with st.spinner("Training model..."):
                trainer.train()
                preds = trainer.predict(data.drop("y", axis=1))

            # See resulting predictions and feature weights
            #st.write("Predictions")
            #st.write(preds)
            st.write("Coefficients")
            st.write(trainer.model.coef_)

        # View data
        # TODO: overlay with a heatmap showing the decisions the model would have made, once the model's trained
        fig = px.scatter(
            data.assign(y=data["y"].astype(str)),
            "x_useful_0",
            "x_useful_1",
            color="y",
        )
        st.plotly_chart(fig)


if __name__ == "__main__":
    # Generate data
    # App should show results that were over-fit, as well as L1 vs. L2 regularised results
    # Also show the weights after each of these 3 scenarios. User can choose which regularisation style is used (or no regularisation), as well as the regularisation strength and maybe the l1 ratio for Elastic Net if it's a mixture of the 2 as I suspect. Are these types of regularisation the same thing as weight decay?
    App.main()
