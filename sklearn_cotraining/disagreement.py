import pandas as pd
from typing import Callable, Tuple
import numpy as np
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from classifiers import CoTrainingClassifier, SeparateViewsClassifier
from data_utils import generate_data, DataGenerationType
import tqdm
from matplotlib import pyplot as plt

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

    if not hasattr(classifier.clf1_, 'predict_proba') or not hasattr(classifier.clf2_, 'predict_proba'):
        raise ValueError("Classifiers must define predict_proba")

    y1 = classifier.clf1_.predict_proba(X[:, :X.shape[1] // 2])[:, 1]
    y2 = classifier.clf2_.predict_proba(X[:, X.shape[1] // 2:])[:, 1]


    return calc_disagreement(y1, y2, d)

def squared_difference(p1: float, p2: float) -> float:
    return (p1 - p2) ** 2

def kl_divergence(p1: float, p2: float) -> float:
    if (p1 == 0).any() or (p1 == 1).any() or (p2 == 0).any() or (p2 == 1).any():
        raise ValueError("Probabilities must be between 0 and 1 (exclusive)")

    kl = p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))
    return kl

def report_disagreement_and_f1(
        classifier: skl.base.BaseEstimator,
        n_samples: int,
        n_features: int,
        n_informative: int,
        prob_replace_background: float=0.,
        prob_invert_class: float=0.,
        random_state: int | None=None,
        gen_type: DataGenerationType=DataGenerationType.SKLEARN

    ) -> Tuple[float, float, float]:
    """
        Takes in a type of classifier, and returns the tuple
        (f1 without cotraining, f1 with separate views, f1 with cotraining, disagreement) obtained by
        training the specified classifiers on a generated dataset
    """
    base_classifier = classifier
    cotrain_classifier = CoTrainingClassifier(clone(classifier), u=1600, p=300, n=300)
    sep_views_classifier = SeparateViewsClassifier(clone(classifier))
    # voting_classifier = VotingCoTraining(clone(classifier), u=1600, p=300, n=300)

    X, y = generate_data(
        n_samples,
        n_features,
        n_informative,
        random_state=random_state,
        prob_replace_background=prob_replace_background,
        prob_invert_class=prob_invert_class,
        gen_type=gen_type
    )

    X_test = X[-n_samples//4:]
    y_test = y[-n_samples//4:]

    X_labeled = X[n_samples//2:-n_samples//4]
    y_labeled = y[n_samples//2:-n_samples//4]

    y = y[:-n_samples//4]
    X = X[:-n_samples//4]

    X1 = X[:,:n_features // 2]
    X2 = X[:, n_features // 2:]

    base_classifier.fit(X_labeled, y_labeled)
    y_pred = base_classifier.predict(X_test)
    base_f1 = skl.metrics.f1_score(y_test, y_pred)

    sep_views_classifier.fit(X1, X2, y)
    y_pred = sep_views_classifier.predict(X_test[:, :n_features // 2], X_test[:, n_features // 2:])
    sep_views_f1 = skl.metrics.f1_score(y_test, y_pred)

    cotrain_classifier.fit(X1, X2, y)
    y_pred = cotrain_classifier.predict(X_test[:, :n_features // 2], X_test[:, n_features // 2:])
    cotrain_f1 = skl.metrics.f1_score(y_test, y_pred)

    voting_classifier.fit(X1, X2, y)
    y_pred = voting_classifier.predict(X_test[:, :n_features // 2], X_test[:, n_features // 2:])
    voting_f1 = skl.metrics.f1_score(y_test, y_pred)

    disagreement = cotrain_disagreement(cotrain_classifier, X, squared_difference)
    return (base_f1, sep_views_f1, cotrain_f1, voting_f1, disagreement)


def main():
    N_SAMPLES = 25000
    N_FEATURES = 1000
    # number of informative and redundant features
    N_INFORMATIVE = N_FEATURES // 100
    random_state = 2

    #probs_replace = np.linspace(0., 0.7, 30)
    probs_replace = np.linspace(0., 0.3, 30)
    progress = tqdm.tqdm(total=len(probs_replace))
    disagreements = []
    base_f1s = []
    sep_views_f1s = []
    cotrain_f1s = []
    voting_f1s = []
    for prob_replace in probs_replace:
        (base_f1, sep_views_f1, cotrain_f1, voting_f1, disagreement) = report_disagreement_and_f1(
            LogisticRegression(max_iter=1000, random_state=random_state),
            N_SAMPLES,
            N_FEATURES,
            N_INFORMATIVE,
            random_state=random_state,
            #prob_replace_background=prob_replace,
            prob_invert_class=prob_replace,
            #gen_type=DataGenerationType.RECTS
            gen_type=DataGenerationType.SKLEARN
        )
        disagreements.append(disagreement)
        base_f1s.append(base_f1)
        sep_views_f1s.append(sep_views_f1)
        cotrain_f1s.append(cotrain_f1)
        voting_f1s.append(voting_f1)
        progress.update(1)

    plt.xlabel("Disagreement")
    plt.ylabel("F1")
    disagreements = np.linspace(0., 0.5, 30)
    plt.scatter(disagreements, base_f1s, label="Base F1")
    plt.scatter(disagreements, sep_views_f1s, label="Separate Views F1")
    plt.scatter(disagreements, cotrain_f1s, label="CoTrain F1")
    plt.scatter(disagreements, voting_f1s, label="Voting F1")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()