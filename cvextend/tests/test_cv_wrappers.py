"""Utility functions for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -

import pandas
import pytest
from imblearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from hmeasure import h_score

from ..cv_wrappers import nested_cv
from ..param_grid import generate_param_grid
from ..score_grid import ScoreGrid

# TODO: test basic_cv


def get_test1_settings():

    pipeline_steps = {
        'preprocessor': {'skip': None},
        'classifier': {
            'svm': SVC(probability=True),
            'rf': RandomForestClassifier()
        }
    }
    params_dict = {
        'skip': {},
        'svm': {'C': [1, 10, 100],
                'gamma': [.01, .1],
                'kernel': ['rbf']},
        'rf': {'n_estimators': [1, 10, 100],
               'max_features': [1, 5, 10, 20]}
    }
    scorer_selection_input = [
        {'score_name': 'H-Measure', 'score_key': 'rank_test_H-Measure',
         'score_criteria': 'min', 'scorer': make_scorer(h_score, needs_proba=True, pos_label=1)},
        {'score_name': 'Accuracy', 'score_key': 'rank_test_Accuracy',
         'score_criteria': 'min', 'scorer': make_scorer(accuracy_score)},
        {'score_name': 'F1-Score', 'score_key': 'rank_test_F1-Score',
         'score_criteria': 'min', 'scorer': make_scorer(f1_score)}
    ]

    pipe = Pipeline([('preprocessor', None), ('classifier', None)])
    param_grid, _ = generate_param_grid(steps=pipeline_steps,
                                        param_dict=params_dict)
    scorer_selection = ScoreGrid(scorer_selection_input)
    
    cv_grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=StratifiedKFold(shuffle=True, n_splits=5),
        scoring=scorer_selection.get_sklearn_dict(),
        return_train_score=True,
        iid=False,
        refit=False,
        verbose=1
    )
    
    random_states = [0, 1]
    outer_cv = StratifiedKFold(n_splits=2)

    X, y = load_breast_cancer(return_X_y=True)

    kwargs = {
        'additional_info': {'data_name': "breast_cancer"},
        'cv_grid': cv_grid,
        'X': X,
        'y': y,
        'inner_cv_seeds': random_states,
        'outer_cv': outer_cv,
        'score_selection': scorer_selection
    }
    return kwargs


def test_nested_cv():
    full_exp_cols = ['data_name', 'estimator', 'inner_cv_random_state', 'outer_fold_n',
                     'param_classifier', 'param_classifier__C', 'param_classifier__gamma',
                     'param_classifier__kernel', 'param_classifier__max_features',
                     'param_classifier__n_estimators', 'param_preprocessor', 'params',
                     'score_name', 'score_value', 'scorer', 'type_classifier',
                     'type_preprocessor']
    # two classifiers, two inner folds and three scores
    exp_min_rows_outer = 2 * 2 * 3

    input_settings = get_test1_settings()

    full_result, _ = nested_cv(**input_settings)

    df_full_result = pandas.DataFrame(full_result)
    assert (df_full_result.columns == full_exp_cols).all()
    assert df_full_result.shape[0] >= exp_min_rows_outer
