import functools
import pandas as pd
from typing import Callable, Tuple
import numpy as np
import sklearn as skl
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from classifiers import CoTrainingClassifier, DistributionAwarePred, DistributionAwareTrain, SeparateViewsClassifier
from data_utils import dom_class, fn_to_mat, generate_data, DataGenerationType, generate_from_probmatrix, random_2class, set_prob_replace_fn, third_class_on_dis
import tqdm
from matplotlib import pyplot as plt

from sklearn_cotraining.utils import calc_accuracy

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
        n_classes: int,
        f: Callable[[int, int, int, int], float]=None,
        mat: np.ndarray=None,
        random_state: int | None=None,

    ) -> Tuple[float, float, float]:
    """
        Takes in a type of classifier, and returns the tuple
        (f1 without cotraining, f1 with separate views, f1 with cotraining, disagreement) obtained by
        training the specified classifiers on a generated dataset
    """
    if not (mat is None):
        n_classes = mat.shape[0]
    prob_tensor = fn_to_mat(f, n_classes) if mat is None else mat
    base_classifier = classifier
    cotrain_classifier = CoTrainingClassifier(clone(classifier), u=1600, p=300, n=300)
    sep_views_classifier = SeparateViewsClassifier(clone(classifier))
    dist_aware_pred = DistributionAwarePred(prob_tensor, clone(classifier), None, p=200, n=200, u=1000, k=40, num_classes=n_classes)
    dist_aware = DistributionAwareTrain(prob_tensor, clone(classifier), None, p=200, n=200, u=1000, k=40, num_classes=n_classes)

    X, y, y1, y2, y_true, y1_true, y2_true = generate_from_probmatrix(
        prob_tensor,
        n_samples,
        n_features,
        n_informative,
        random_state=random_state,
        two_labels=True
    )
    '''
    X, y = generate_data(
        n_samples,
        n_features,
        n_informative,
        random_state=random_state,
        prob_replace_background=prob_replace_background,
        prob_invert_class=prob_invert_class,
        gen_type=gen_type
    )
    '''
    print('----------------')
    num_corr = 0
    for i in range(len(y_true)):
        if y_true[i] == y1_true[i] and y_true[i] == y2_true[i]:
            num_corr += 1
        elif y1_true[i] != y2_true[i]:
            num_corr += 0.5
    print('best possible acc', num_corr / len(y_true))
    print('----------------')

    X_test = X[-n_samples//4:]
    y_test = y[-n_samples//4:]
    y1_test = y1_true[-n_samples//4:]
    y2_test = y2_true[-n_samples//4:]

    X_labeled = X[n_samples//2:-n_samples//4]
    y_labeled = y[n_samples//2:-n_samples//4]

    y = y[:-n_samples//4]
    y1 = y1[:-n_samples//4]
    y2 = y2[:-n_samples//4]
    X = X[:-n_samples//4]

    X1 = X[:,:n_features // 2]
    X2 = X[:, n_features // 2:]

    base_classifier.fit(X_labeled, y_labeled)
    y_pred = base_classifier.predict(X_test)
    base_acc = calc_accuracy(base_classifier, X_test)
    base_f1 = score(y_test, y_pred)[2].mean()

    sep_views_classifier.fit(X1.copy(), X2.copy(), y.copy())
    y_pred = sep_views_classifier.predict(X_test[:, :n_features // 2], X_test[:, n_features // 2:])
    sep_views_f1 = score(y_test, y_pred)[2].mean()

    cotrain_classifier.fit(X1.copy(), X2.copy(), y.copy(), y1_true.copy(), y2_true.copy(), y_true.copy())
    y_pred = cotrain_classifier.predict(X_test[:, :n_features // 2], X_test[:, n_features // 2:])
    cotrain_f1 = score(y_test, y_pred)[2].mean()

    dist_aware_pred.fit(X1.copy(), X2.copy(), y.copy())
    y_pred = dist_aware_pred.predict(X_test[:, :n_features // 2], X_test[:, n_features // 2:])
    dist_aware_pred_f1 = score(y_test, y_pred)[2].mean()

    dist_aware.fit(X1.copy(), X2.copy(), y1.copy(), y2.copy(), y1_true.copy(), y2_true.copy(), y_true.copy())
    y_pred = dist_aware.predict(X_test[:, :n_features // 2], X_test[:, n_features // 2:])
    dist_aware_f1 = score(y_test, y_pred)[2].mean()

    disagreement = cotrain_disagreement(cotrain_classifier, X, squared_difference)
    print('base_f1', base_f1)
    print('sep_views_f1', sep_views_f1)
    print('cotrain_f1', cotrain_f1)
    print('dist_aware_pred_f1', dist_aware_pred_f1)
    print('dist_aware_f1', dist_aware_f1)
    return (base_f1, sep_views_f1, cotrain_f1, dist_aware_pred_f1, dist_aware_f1, disagreement)


def main():
    N_SAMPLES = 25000
    N_FEATURES = 1000
    # number of informative and redundant features
    N_INFORMATIVE = N_FEATURES // 100
    NUM_RANDOM_STATES = 5
    random_states = [i for i in range(NUM_RANDOM_STATES)]

    #probs_replace = np.linspace(0., 0.7, 30)
    probs_replace = np.linspace(0.3, 1., 30)
    progress = tqdm.tqdm(total=len(probs_replace) * NUM_RANDOM_STATES)
    disagreements = []
    base_f1s = []
    sep_views_f1s = []
    cotrain_f1s = []
    dist_aware_pred_f1s = []
    dist_aware_f1s = []
    for prob_replace in probs_replace:
        disagreements.append(0)
        base_f1s.append(0)
        sep_views_f1s.append(0)
        cotrain_f1s.append(0)
        dist_aware_f1s.append(0)
        dist_aware_pred_f1s.append(0)
        for random_state in random_states:
            dis_and_f1 = report_disagreement_and_f1(
                LogisticRegression(max_iter=1000, random_state=random_state),
                N_SAMPLES,
                N_FEATURES,
                N_INFORMATIVE,
                n_classes=3,
                #f=functools.partial(third_class_on_dis, p_diag=prob_replace),
                mat = random_2class(prob_replace),
                random_state=random_state,
            )
            print(dis_and_f1)
            (base_f1, sep_views_f1, cotrain_f1, dist_aware_pred_f1, dist_aware_f1, disagreement) = dis_and_f1
            disagreements[-1] += disagreement / NUM_RANDOM_STATES
            base_f1s[-1] += base_f1 / NUM_RANDOM_STATES
            sep_views_f1s[-1] += sep_views_f1 / NUM_RANDOM_STATES
            cotrain_f1s[-1] += cotrain_f1 / NUM_RANDOM_STATES
            dist_aware_pred_f1s[-1] += dist_aware_pred_f1 / NUM_RANDOM_STATES
            dist_aware_f1s[-1] += dist_aware_f1 / NUM_RANDOM_STATES
            progress.update(1)

    plt.xlabel("Agreement")
    plt.ylabel("F1")
    disagreements = probs_replace
    plt.scatter(disagreements, base_f1s, label="Base F1")
    plt.scatter(disagreements, sep_views_f1s, label="Separate Views F1")
    plt.scatter(disagreements, cotrain_f1s, label="CoTrain F1")
    plt.scatter(disagreements, dist_aware_pred_f1s, label="Dist Aware Pred F1")
    plt.scatter(disagreements, dist_aware_f1s, label="Dist Aware Train + Pred F1")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()