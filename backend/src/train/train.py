"""
Программа: Тренировка данных
Версия: 1.0 Pymagic
"""

import optuna
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from optuna.integration import LightGBMPruningCallback


from optuna import Study

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics

def objective_lgb(trial, X, y, random_state=10):
   
    
    lgb_params = {
        
        """
        Признаки для бинарной классификации с помощью алгоритма LGBM
        Целевая функция для поиска параметров
        :param trial: кол-во trials
        :param X: данные объект-признаки
        :param y: данные с целевой переменной
        :param n_folds: кол-во фолдов
        :param random_state: random_state
        :return: среднее значение метрики по фолдам
        """
        "n_estimators":
        trial.suggest_categorical("n_estimators", [700]),
        "verbosity": -1,
        "learning_rate":
        trial.suggest_categorical("learning_rate", [0.1]),
        "num_class": 1,
        "num_leaves":
        trial.suggest_int("num_leaves", 66, 66),
        "max_depth":
        trial.suggest_int("max_depth", 9, 9),
        "min_child_samples":
        trial.suggest_int("min_child_samples", 7, 7),
        "reg_alpha":
        trial.suggest_float("reg_alpha", 0.0038, 0.004, log=True),
        "gamma":
        trial.suggest_int("gamma", 2, 2),
        "reg_lambda":
        trial.suggest_float("reg_lambda", 4.8, 5.06, log=True),
        "min_split_gain":
        trial.suggest_int("min_split_gain", 1, 1),
        "subsample":
        trial.suggest_float("subsample", 0.8, 0.9),
        "subsample_freq":
        trial.suggest_categorical("subsample_freq", [1]),
        "colsample_bytree":
        trial.suggest_float("colsample_bytree", 0.98, 1.0),
        "objective":
        trial.suggest_categorical("objective", ["binary"]),
        "random_state":
        trial.suggest_categorical("random_state", [10]),
        "boosting_type":
        "gbdt",
        "lambda_l1":
        trial.suggest_float("lambda_l1", 0.13, 0.18, log=True),
        "lambda_l2":
        trial.suggest_float("lambda_l2", 0.0095, 0.014, log=True),
        "feature_fraction":
        trial.suggest_float("feature_fraction", 0.78, 0.8),
    }
    
    
    N_FOLDS = 3
    cv = KFold(n_splits=N_FOLDS, shuffle=True)
    cv_predicts = np.empty(N_FOLDS)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    dtrain = lgb.Dataset(X_train, label=y_train, )
    train_labels = dtrain.get_label()
    ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1)
    
    # параметр для стрижки деревьев алгоритма LGBMClassificier
    pruning_callback = LightGBMPruningCallback(trial, "binary_logloss")
    model = LGBMClassifier(**lgb_params,
                           early_stopping_rounds=100,
                           scale_pos_weight=ratio)
    
    
    # обучаем модель                       
    model.fit(X_train,
              y_train,
              eval_metric="auc",
              eval_set=[(X_test, y_test)],
              callbacks=[pruning_callback,
                         lgb.early_stopping(100)])

    preds = model.predict(X_test)
    cv_predicts[idx] =  roc_auc_score(y_test, preds)
    return np.mean(cv_predicts)


def find_optimal_params(
    data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data_train: датасет train
    :param data_test: датасет test
    :return: [LGBMClassifier tuning, Study]
    """
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=kwargs["target_column"]
    )

    study = optuna.create_study(direction="minimize", study_name="LGB")
    function = lambda trial: objective_lgb(
        trial, x_train, y_train, kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)
    return study


def train_model(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    study: Study,
    target: str,
    metric_path: str,
) -> LGBMClassifier:
    """
    Обучение модели на лучших параметрах
    :param data_train: тренировочный датасет
    :param data_test: тестовый датасет
    :param study: study optuna
    :param target: название целевой переменной
    :param metric_path: путь до папки с метриками
    :return: LGBMClassifier
    """
    # get data
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=target
    )

    # training optimal params
    clf = LGBMClassifier(**study.best_params, verbose=-1)
    clf.fit(x_train, y_train)
 
    
    # save metrics
    x_test['Birthyear'] = np.exp(x_test['Birthyear']) - 1
    x_test['Tenure'] = np.exp(x_test['Tenure'])
    save_metrics(data_x=x_test, data_y=y_test, model=clf, metric_path=metric_path)
    return clf
