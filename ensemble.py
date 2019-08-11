import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.externals.joblib import Parallel


class ModelEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 assembly_estimator,
                 intermediate_estimators,
                 ensemble_train_size=.25,
                 n_jobs=1):
        self.assembly_estimator = assembly_estimator
        self.intermediate_estimators = intermediate_estimators
        self.ensemble_train_size = ensemble_train_size
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from sklearn.model_selection import train_test_split

        if self.ensemble_train_size == 1:
            X_train, y_train = X, y
            X_holdout, y_holdout = X, y
        else:
            splits = train_test_split(X, y, train_size=self.ensemble_train_size)
            X_train, X_holdout, y_train, y_holdout = splits

        Parallel(n_jobs=self.n_jobs)(
            ((fit_est, [est, X_train, y_train], {}) for est in self.intermediate_estimators)
        )

        probas = np.vstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_est, [est, X_holdout], {}) for est in self.intermediate_estimators)
        )).T

        self.assembly_estimator.fit(probas, y_holdout)

        return self

    def predict(self, X):
        probas = np.vstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_est, [est, X], {}) for est in self.intermediate_estimators)
        )).T
        return self.assembly_estimator.predict(probas)


def predict_est(estimator, features):
    return estimator.predict(features)
