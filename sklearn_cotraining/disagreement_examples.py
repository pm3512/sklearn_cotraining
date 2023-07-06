import functools
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from classifiers import CoTrainingClassifier, DistributionAwarePred, SeparateViewsClassifier
from disagreement import cotrain_disagreement, kl_divergence, squared_difference
from data_utils import DataGenerationType, dom_class, generate_data, generate_from_probmatrix, fn_to_mat, identity_fn
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    N_SAMPLES = 25000
    N_FEATURES = 1000
    # number of informative and redundant features
    num_classes = 2
    N_INFORMATIVE = N_FEATURES // 100
    generator_tensor = fn_to_mat(functools.partial(dom_class, p_100=0.05), num_classes)

    X, y = generate_from_probmatrix(
        generator_tensor,
        N_SAMPLES,
        N_FEATURES,
        N_INFORMATIVE,
        random_state=12,
    )

    
    '''
    X, y = generate_data(
        N_SAMPLES,
        N_FEATURES,
        N_INFORMATIVE,
        random_state=12,
    )
    '''

    X_test = X[-N_SAMPLES//4:]
    y_test = y[-N_SAMPLES//4:]

    X_labeled = X[N_SAMPLES//2:-N_SAMPLES//4]
    y_labeled = y[N_SAMPLES//2:-N_SAMPLES//4]

    #plt.scatter(X_labeled[:, 0], X_labeled[:, 1], c=y_labeled)
    #plt.show()

    y = y[:-N_SAMPLES//4]
    X = X[:-N_SAMPLES//4]

    X1 = X[:,:N_FEATURES // 2]
    X2 = X[:, N_FEATURES // 2:]

    print('Logistic')
    base_lr = LogisticRegression(max_iter=1000)
    base_lr.fit(X_labeled, y_labeled)
    y_pred = base_lr.predict(X_test)
    print(classification_report(y_test, y_pred))

    print ('Logistic Separate View')
    sep_view = SeparateViewsClassifier(LogisticRegression(max_iter=1000))
    sep_view.fit(X1, X2, y)
    y_pred = sep_view.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
    print (classification_report(y_test, y_pred))
    print('Disagreement: ', cotrain_disagreement(sep_view, X, squared_difference))

    print ('Logistic CoTraining')
    lg_co_clf = CoTrainingClassifier(LogisticRegression(max_iter=1000), u=1000, p=200, n=200, num_classes=num_classes)
    lg_co_clf.fit(X1, X2, y)
    y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
    print (classification_report(y_test, y_pred))
    print('Disagreement: ', cotrain_disagreement(lg_co_clf, X, squared_difference))

    print ('Logistic distribution aware prediction')
    dist_aware = DistributionAwarePred(prob_tensor=generator_tensor, clf=LogisticRegression(max_iter=1000), clf2=None, u=1000, p=200, n=200, k=40, num_classes=num_classes)
    dist_aware.fit(X1, X2, y)
    y_pred = dist_aware.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
    print (classification_report(y_test, y_pred))
    print('Disagreement: ', cotrain_disagreement(dist_aware, X, squared_difference))
