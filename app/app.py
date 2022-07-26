from itertools import product
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import streamlit as st
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore

L1 = "L1"
L2 = "L2"
MODEL_PARAM_MAP = {L1: "l1", L2: "l2"}
N_USEFUL_FEATURES = 2

DV = "y"

# TODO: nearly due for a big refactor


def true_func(x: np.ndarray) -> np.ndarray:
    """
    This cuts a diagonal line through feature space
    Built on the assumption that there are 2 useful features
    """
    y = x.sum(axis=1) > sum(DataCreator.IV_RANGE)
    return y.astype(int)


# TODO: is it recalculating more frequently than it needs to?
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
        n_noise_features: int,
        noise_amount: float,
        function: Callable[[np.ndarray], np.ndarray] = true_func,
    ) -> pd.DataFrame:
        np.random.seed(777)

        if not (0 <= noise_amount <= 1):
            raise ValueError

        useful_features = np.random.uniform(
            *cls.IV_RANGE,
            size=(cls.N_SAMPLES, N_USEFUL_FEATURES),
        )
        labels = function(useful_features).reshape(-1, 1)

        # After getting a label based on the "true" value, add error
        useful_features += np.random.uniform(
            -noise_amount,
            noise_amount,
            size=(cls.N_SAMPLES, N_USEFUL_FEATURES),
        )
        useful_features = useful_features.clip(*cls.IV_RANGE)

        noise_features = np.random.uniform(
            *cls.IV_RANGE,
            size=(cls.N_SAMPLES, n_noise_features),
        )

        df = pd.DataFrame(
            np.concatenate((useful_features, noise_features), axis=1),
            columns=(
                *(f"x_useful_{i}" for i in range(N_USEFUL_FEATURES)),
                *(f"x_noise_{i}" for i in range(n_noise_features)),
            ),
        ).assign(y=labels)

        return df


class Trainer:
    def __init__(
        self,
        model: BaseEstimator,
        data: pd.DataFrame,
    ) -> None:
        self.X = data[[col for col in data if col != DV]]
        self.y = data[DV]
        self.model = model
        self._trained = False

    def train(self) -> None:
        self.model.fit(self.X, self.y)
        self._trained = True

    def predict(self, x: pd.DataFrame) -> List[int]:
        assert self._trained
        return self.model.predict(x).tolist()

    def get_heatmap(
        self,
        data: Dict[str, Union[Sequence[float], np.ndarray]],
    ) -> pd.DataFrame:
        assert self._trained
        df = pd.DataFrame(product(*data.values()), columns=data.keys())
        df[DV] = self.predict(df)
        return df


def get_model_coefficients_df(
    model: BaseEstimator, data: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(model.coef_, columns=data.drop(DV, axis=1).columns)


class App:
    @staticmethod
    def main() -> None:
        st.header("L1 vs. L2 Regularisation Intuition")

        # Create data
        data = DataCreator.create_data(2, 0.5, true_func)

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

        if model_trained or True:
            with st.spinner("Training model..."):
                trainer.train()
                # preds = trainer.predict(data.drop(DV, axis=1))

            # See resulting predictions and feature weights
            st.write("Coefficients")
            st.write(get_model_coefficients_df(trainer.model, data))

            # View data
            resolution = 100
            heatmap = trainer.get_heatmap(
                {
                    "x_useful_0": np.linspace(
                        *DataCreator.IV_RANGE, num=resolution
                    ),
                    "x_useful_1": np.linspace(
                        *DataCreator.IV_RANGE, num=resolution
                    ),
                    "x_noise_0": [0],
                    "x_noise_1": [0],
                }
            )

            heatmap_negative_colour = "rgba(255, 0, 0, 0.5)"
            heatmap_positive_colour = "rgba(0, 255, 0, 0.5)"
            prediction_negative_colour = "rgb(255, 0, 0)"
            prediction_positive_colour = "rgb(0, 255, 0)"

            fig = px.scatter(
                data.assign(y=data[DV].astype(str)),
                "x_useful_0",
                "x_useful_1",
                color=DV,
                color_discrete_map={
                    "0": prediction_negative_colour,
                    "1": prediction_positive_colour,
                },
            )

            heatmap_plot_data = (
                heatmap[["x_useful_0", "x_useful_1", DV]]
                .drop_duplicates()
                .pivot("x_useful_0", "x_useful_1", DV)
            )

            heatmap_fig = px.imshow(
                heatmap_plot_data.values,
                x=heatmap_plot_data.index,
                y=heatmap_plot_data.columns,
                color_continuous_scale=[
                    (0, heatmap_negative_colour),
                    (1, heatmap_positive_colour),
                ],
            )

            heatmap_fig.add_trace(fig.data[0])
            heatmap_fig.add_trace(fig.data[1])
            heatmap_fig.update_layout(coloraxis_showscale=False)

            st.plotly_chart(heatmap_fig)


if __name__ == "__main__":
    # Generate data
    # App should show results that were over-fit, as well as L1 vs. L2
    # regularised results
    # Also show the weights after each of these 3 scenarios. User can choose
    # which regularisation style is used (or no regularisation), as well as the
    # regularisation strength and maybe the l1 ratio for Elastic Net if it's a
    # mixture of the 2 as I suspect. Are these types of regularisation the same
    # thing as weight decay?
    App.main()
