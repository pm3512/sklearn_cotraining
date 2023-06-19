# code for hyperparameter optimization
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from classifiers import CoTrainingClassifier
from data_utils import DataGenerationType, generate_data

def optimize_fn(trial: optuna.Trial):
    N_SAMPLES = 25000
    N_FEATURES = 1000
    # number of informative and redundant features
    N_INFORMATIVE = N_FEATURES // 100
    X, y = generate_data(
        N_SAMPLES,
        N_FEATURES,
        N_INFORMATIVE,
        random_state=1,
        prob_replace=0.,
        permute_cols=True,
        gen_type=DataGenerationType.SKLEARN
    )

    X_test = X[-N_SAMPLES//4:]
    y_test = y[-N_SAMPLES//4:]

    y = y[:-N_SAMPLES//4]
    X = X[:-N_SAMPLES//4]

    X1 = X[:,:N_FEATURES // 2]
    X2 = X[:, N_FEATURES // 2:]
    
    u = trial.suggest_int('u', 100, 10000)
    pn = trial.suggest_int('p', 2, 300)
    try:
        lg_co_clf = CoTrainingClassifier(LogisticRegression(max_iter=1000), u=u, p=pn, n=pn)
        lg_co_clf.fit(X1, X2, y)
        y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
        return f1_score(y_test, y_pred)
    except:
        return 0
if __name__ == '__main__':

    # optimize
    study = optuna.create_study(direction='maximize')
    optimize = lambda trial: optimize_fn(trial)
    study.optimize(optimize, n_trials=300)
    print(study.best_params)