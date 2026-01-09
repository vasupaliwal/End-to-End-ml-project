from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_openml


@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series


def load_ames_housing() -> DatasetBundle:
    """Load the Ames Housing dataset from OpenML.

    Returns:
        DatasetBundle containing features and target (SalePrice).
    """
    dataset = fetch_openml(name="house_prices", as_frame=True)
    frame = dataset.frame
    if frame is None:
        raise ValueError("OpenML did not return a dataframe.")

    target = frame["SalePrice"].astype(float)
    features = frame.drop(columns=["SalePrice"])
    return DatasetBundle(features=features, target=target)
