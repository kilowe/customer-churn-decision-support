from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from preprocessing import run_preprocessing


RANDOM_STATE = 42
TEST_SIZE = 0.20


def predict_with_threshold(
    model: LogisticRegression, X: pd.DataFrame, threshold=float
) -> np.ndarray:

    proba_churn = model.predict_proba(X)[:, 1]
    return (proba_churn > threshold).astype(int)
