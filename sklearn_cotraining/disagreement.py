import pandas as pd
from typing import Callable
import numpy as np
from classifiers import CoTrainingClassifier

def calc_disagreement(y1: pd.Series, y2: pd.Series, d: Callable[[float, float], float]) -> float:
    if len(y1) != len(y2):
        raise ValueError("Label series lengths do not match")

    result = d(y1, y2)
    return result.mean()

def cotrain_disagreement(
        classifier: CoTrainingClassifier,
        X: pd.DataFrame,
        d: Callable[[float, float], float]
    ) -> float:
    y1 = classifier.clf1_.predict(X[:, :X.shape[1] // 2])
    y2 = classifier.clf2_.predict(X[:, X.shape[1] // 2:])

    return calc_disagreement(y1, y2, d)

def squared_difference(p1: float, p2: float) -> float:
    return (p1 - p2) ** 2

def kl_divergence(p1: float, p2: float) -> float:
    if (p1 == 0).any() or (p1 == 1).any() or (p2 == 0).any() or (p2 == 1).any():
        raise ValueError("Probabilities must be between 0 and 1 (exclusive)")

    kl = p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))
    return kl

if __name__ == '__main__':
    series1 = pd.Series([0.2, 0.3, 0.4, 0.5])
    series2 = pd.Series([0.8, 0.1, 0.3, 0.8])

    average_diff = calc_disagreement(series1, series2, kl_divergence)
    print(average_diff)
