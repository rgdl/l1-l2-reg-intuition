import numpy as np
import pandas as pd  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore

from app.app import DV
from app.app import DataCreator
from app.app import Trainer
from app.app import get_model_coefficients_df


def test_create_data_dtypes():
    df = DataCreator().create_data(2, 0.5)

    print(df.dtypes)
    x_dtypes = df[[col for col in df if col.startswith("x_")]].dtypes
    assert x_dtypes.nunique() == 1
    assert "float" in str(x_dtypes.iloc[0])

    y_dtype = df.dtypes[DV]
    assert "int" in str(y_dtype)


def test_create_data_value_ranges():
    df = DataCreator().create_data(2, 0.5)

    x_vals = df[[col for col in df if col.startswith("x_")]].values
    assert x_vals.min() >= DataCreator.IV_RANGE[0]
    assert x_vals.max() <= DataCreator.IV_RANGE[1]

    y_vals = df[DV]
    assert y_vals.isin({0, 1}).all()


def test_create_data_roughly_balanced_classes():
    """
    WARNING: this is a flaky test, as it depends on probabilistic functions
    However, it should pass most of the time
    """
    df = DataCreator().create_data(2, 0.5)

    assert df[DV].value_counts(normalize=True).max() < 0.6


def test_trainer():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 5, 1, 0],
        }
    )
    df[DV] = (df["A"] < df["B"]).astype(int)
    trainer = Trainer(LogisticRegression(), df)
    trainer.train()
    preds = trainer.predict(df[["A", "B"]])
    assert preds == [1, 1, 0, 0]
    assert hasattr(trainer.model, "coef_")


def test_get_model_coefficients_df():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 5, 1, 0],
        }
    )
    df[DV] = (df["A"] < df["B"]).astype(int)
    model = LogisticRegression()
    trainer = Trainer(model, df)
    trainer.train()
    coefs = get_model_coefficients_df(model, df)
    assert list(coefs.columns) == ["A", "B"]
    assert len(coefs) == 1


def test_get_heatmap():
    """
    Create a space with a very obvious boundary
    """
    df = pd.DataFrame(
        {
            "A": [0 for _ in range(20)],
            "B": range(20),
        }
    )
    df[DV] = (df["B"] > 10).astype(int)
    model = LogisticRegression()
    trainer = Trainer(model, df)
    trainer.train()
    heatmap = trainer.get_heatmap(
        {"A": np.linspace(0, 20, 10), "B": np.linspace(0, 2, 5)}
    )
    assert heatmap.shape == (50, 3)
    assert all(heatmap.loc[heatmap["B"] < 5, DV] == 0)
    assert all(heatmap.loc[heatmap["B"] > 15, DV] == 1)
