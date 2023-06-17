from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from classifiers import CoTrainingClassifier, SeparateViewsClassifier
from disagreement import cotrain_disagreement, kl_divergence, squared_difference
from data_utils import generate_data
import numpy as np


if __name__ == '__main__':
    N_SAMPLES = 25000
    N_FEATURES = 1000
    # number of informative and redundant features
    N_INFORMATIVE = N_FEATURES // 100
    X, y = generate_data(N_SAMPLES, N_FEATURES, N_INFORMATIVE, random_state=1, prob_replace=0.1)

    X_test = X[-N_SAMPLES//4:]
    y_test = y[-N_SAMPLES//4:]

    X_labeled = X[N_SAMPLES//2:-N_SAMPLES//4]
    y_labeled = y[N_SAMPLES//2:-N_SAMPLES//4]

    y = y[:-N_SAMPLES//4]
    X = X[:-N_SAMPLES//4]

    X1 = X[:,:N_FEATURES // 2]
    X2 = X[:, N_FEATURES // 2:]

    print('num positive: ', np.sum(y == 1))
    print('Logistic')
    base_lr = LogisticRegression(max_iter=100000)
    base_lr.fit(X_labeled, y_labeled)
    y_pred = base_lr.predict(X_test)
    print(classification_report(y_test, y_pred))

    print ('Logistic Separate View')
    sep_view = SeparateViewsClassifier(LogisticRegression(max_iter=100000))
    sep_view.fit(X1, X2, y)
    y_pred = sep_view.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
    print (classification_report(y_test, y_pred))
    print('Disagreement: ', cotrain_disagreement(sep_view, X, squared_difference))

    print ('Logistic CoTraining')
    lg_co_clf = CoTrainingClassifier(LogisticRegression(max_iter=100000))
    lg_co_clf.fit(X1, X2, y)
    y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
    print (classification_report(y_test, y_pred))
    print('Disagreement: ', cotrain_disagreement(lg_co_clf, X, squared_difference))
