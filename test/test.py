import pandas as pd  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore

from app.app import DataCreator  # type: ignore
from app.app import Trainer  # type: ignore


def test_create_data_dtypes():
    df = DataCreator().create_data(2, 2, 0.5)

    print(df.dtypes)
    x_dtypes = df[[col for col in df if col.startswith("x_")]].dtypes
    assert x_dtypes.nunique() == 1
    assert "float" in str(x_dtypes.iloc[0])

    y_dtype = df.dtypes["y"]
    assert "int" in str(y_dtype)


def test_create_data_value_ranges():
    df = DataCreator().create_data(2, 2, 0.5)

    x_vals = df[[col for col in df if col.startswith("x_")]].values
    # TODO: this should be determined by function arguments, not some constant!
    assert x_vals.min() >= -1
    assert x_vals.max() <= 1

    y_vals = df["y"]
    assert y_vals.isin({0, 1}).all()


def test_create_data_roughly_balanced_classes():
    """
    WARNING: this is a flaky test, as it depends on probabilistic functions
    However, it should pass most of the time
    """
    df = DataCreator().create_data(2, 2, 0.5)

    assert df["y"].value_counts(normalize=True).max() < 0.6


def test_trainer():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 5, 1, 0],
        }
    )
    df["y"] = (df["A"] < df["B"]).astype(int)
    trainer = Trainer(LogisticRegression(), df, dv="y")
    trainer.train()
    preds = trainer.predict(df[["A", "B"]])
    assert preds == [1, 1, 0, 0]
    assert hasattr(trainer.model, "coef_")
