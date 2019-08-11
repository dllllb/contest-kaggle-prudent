import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import fmin_powell
from xgboost import XGBRegressor

from ensemble import ModelEnsembleRegressor
from qwk import quadratic_weighted_kappa


def update_model_stats(stats_file, params, results):
    import json
    import os.path
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []
        
    stats.append({**results, **params})
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

        
def run_experiment(evaluator, params, stats_file):    
    import time
    
    params = init_params(params)
    start = time.time()
    scores = evaluator(params)
    exec_time = time.time() - start
    update_model_stats(stats_file, params, {**scores, 'exec-time-sec': exec_time})


def preds_to_rank(preds, min, max):
    return np.clip(np.round(preds), min, max).astype(int)


def qwk_score(est, features, labels):
    raw_pred = est.predict(features)
    pred = preds_to_rank(raw_pred, np.min(labels), np.max(labels))
    return quadratic_weighted_kappa(labels, pred)


def score_offset(pred_base, pred_modified, labels, offset, rank_number, scorer):
    pred_modified[pred_base.astype(int) == rank_number] = pred_base[pred_base.astype(int) == rank_number] + offset
    rank = preds_to_rank(pred_modified, np.min(labels), np.max(labels))
    score = scorer(labels, rank)
    return score


def apply_offsets(data, offsets):
    res = np.copy(data)
    for j in range(len(offsets)):
        res[data.astype(int) == j] = data[data.astype(int) == j] + offsets[j]
    return res


def minimize_reminders(preds, true, scorer):
    offsets = np.zeros(len(set(true)))
    optimized_preds = apply_offsets(preds, offsets)
    for j in range(len(offsets)):
        def train_offset(x): return -score_offset(preds, optimized_preds, true, x, j, scorer) * 100
        offsets[j] = fmin_powell(train_offset, offsets[j])
    return offsets


class RemindersMinimizingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, scorer):
        self.base_estimator = base_estimator
        self.scorer = scorer
        self.offsets = None

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        preds = self.base_estimator.predict(X)

        self.offsets = minimize_reminders(preds, y, self.scorer)

        return self

    def predict(self, X):
        preds = self.base_estimator.predict(X)
        preds_fix = apply_offsets(preds, self.offsets)
        return preds_fix


def cv_test(est, n_folds):
    df = pd.read_csv('train.csv.gz', index_col='Id')
    features = df.drop(['Response'], axis=1)
    target = df['Response'].values
    scores = cross_val_score(
        estimator=est,
        X=features,
        y=target,
        cv=StratifiedKFold(n_folds, shuffle=True),
        scoring=qwk_score,
        n_jobs=1,
        verbose=0)
    return {'qwk-mean': scores.mean(), 'qwk-std': scores.std()}


def submission(est, name='results'):
    df = pd.read_csv('train.csv.gz', index_col='Id')
    features = df.drop(['Response'], axis=1)
    target = df['Response'].values
    model = est.fit(features, target)

    df_test = pd.read_csv('test.csv.gz', index_col='Id')

    y_pred = preds_to_rank(model.predict(df_test), np.min(target), np.max(target))

    res = pd.Series(y_pred, index=df_test.index, name='Response')
    res.to_csv(name+'.csv', index_label='Id', header=True)
    
    
def pred_vs_true(est, path):
    df = pd.read_csv('train.csv.gz', index_col='Id')
    features = df.drop(['Response'], axis=1)
    target = df['Response'].values

    x_train, x_test, y_train, y_test = train_test_split(features, target, train_size=0.9)
    y_pred = est.fit(x_train, y_train).predict(x_test)

    pd.DataFrame({'pred': y_pred, 'true': y_test}).to_csv(path, index=False, sep='\t')
    
    
def results_corr(pred_files):
    preds = np.array(
        [pd.read_csv(file, index_col='Id', squeeze=True) for file in pred_files],
        dtype=np.int32)
    print(pd.DataFrame(np.corrcoef(preds), index=pred_files, columns=pred_files))
    
    
def submission_mix(pred_files, name):
    preds = pd.DataFrame({
        file: pd.read_csv(file, index_col='Id', squeeze=True)
        for file in pred_files})
    mix = preds_to_rank(preds.mean(axis=1), 1, 8)
    res = pd.Series(mix, index=preds.index, name='Response')
    res.to_csv(name+'.csv', index_label='Id', header=True)
    
    
def submission_mix_m(pred_files, name):
    preds = pd.DataFrame({
        file: pd.read_csv(file, index_col='Id', squeeze=True)
        for file in pred_files})
    mix = preds.mode(axis=1)[0]
    mix[mix.isnull()] = preds.median(axis=1)
    res = pd.Series(mix.astype(int), name='Response')
    res.to_csv(name+'.csv', index_label='Id', header=True)


def init_params(overrides):
    defaults = {
        'validation-type': 'cv',
        'n_folds': 3,
        'rmin': False,
    }
    return {**defaults, **overrides}


def init_xgb_est(params):
    keys = {
        'learning_rate',
        'n_estimators',
        'max_depth',
        'min_child_weight',
        'gamma',
        'subsample',
        'colsample_bytree',
    }
    
    xgb_params = {
        "objective": "reg:linear",
        "scale_pos_weight": 1,
        **{k: v for k, v in params.items() if k in keys},
    }

    class XGBC(XGBRegressor):
        def fit(self, x, y, **kwargs):
            f_train, f_val, t_train, t_val = train_test_split(x, y, test_size=params['es_share'])
            super().fit(
                f_train,
                t_train,
                eval_set=[(f_val, t_val)],
                early_stopping_rounds=params['num_es_rounds'],
                verbose=params['num_es_rounds'])

    return XGBC(**xgb_params)


def validate(params):
    df2dict = FunctionTransformer(
        lambda x: x.to_dict(orient='records'), validate=False)

    transf = make_pipeline(
        df2dict,
        DictVectorizer(sparse=False),
        SimpleImputer(strategy='median'),
    )
    
    est_type = params['est_type']
    if est_type == 'xgb':
        est = init_xgb_est(params)
    elif est_type == 'rfr':
        est = RandomForestRegressor(n_estimators=params['n_estimators'], n_jobs=-1, verbose=1)
    elif est_type == 'etree':
        est = ExtraTreesRegressor(n_estimators=params['n_estimators'], n_jobs=-1, verbose=1)
    elif est_type == 'xgb/dt':
        est = ModelEnsembleRegressor(
            intermediate_estimators=[
                init_xgb_est(params),
            ],
            assembly_estimator=DecisionTreeClassifier(max_depth=2),
            ensemble_train_size=1
        )
    else:
        raise AssertionError(f'unknown estimator type: {est_type}')
        
    if params['rmin']:
        est = RemindersMinimizingRegressor(
            base_estimator=est,
            scorer=quadratic_weighted_kappa
        )
    
    est = make_pipeline(transf, est)
    return cv_test(est, n_folds=params['n_folds'])


def test_validate():
    params = {
        "eta": 0.1,
        "min_child_weight": 6,
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        "max_depth": 6,
        "num_rounds": 10,
        "num_es_rounds": 120,
        "es_share": .05,
        "est_type": "xgb",
    }
    print(validate(init_params(params)))
