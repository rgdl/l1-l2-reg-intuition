from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

L1 = 'L1'
L2 = 'L2'


def true_func(x):
    y = (x ** 2).sum(1) ** 0.5 < 0.8
    return y.astype(int)


# TODO: replace this class with a function if it only has one method
class DataCreator:
    """
    We want a dataset which is:
    * Easy to visualise: maybe a binary classifier over 2 useful input features, along with 2 non-diagnostic features which we don't have to plot
    * Noisy enough that over-fitting is possible (random swaps of the training labels)

    Note that over-fitting requires a level of complexity in the model also. This might be easiest to achieve with a neural network.

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
                *(f'x_useful_{i}' for i in range(n_useful_features)),
                *(f'x_noise_{i}' for i in range(n_noise_features)),
            )
        ).assign(y=labels)

        return df


class App:
    @staticmethod
    def main() -> None:
        st.header('L1 vs. L2 Regularisation Intuition')

        # Create data
        data = DataCreator.create_data(2, 2, 0, true_func)
        st.write(data)

        # Preview data
        fig = px.scatter(data.assign(y=data['y'].astype(str)), 'x_useful_0', 'x_useful_1', color='y')
        st.plotly_chart(fig)

        # Select Regularisation Option
        st.selectbox('Choose a regularisation type', (L1, L2))

        # Select Regularisation Strength and other things as appropriate
        st.write('TODO: Select regularisation strength, other parameters?')

        # See resulting predictions and feature weights
        st.write('TODO: See resulting predictions and feature weights')


if __name__ == '__main__':
    # Generate data
    # App should show results that were over-fit, as well as L1 vs. L2 regularised results
    # Also show the weights after each of these 3 scenarios. User can choose which regularisation style is used (or no regularisation), as well as the regularisation strength and maybe the l1 ratio for Elastic Net if it's a mixture of the 2 as I suspect. Are these types of regularisation the same thing as weight decay?
    App.main()
