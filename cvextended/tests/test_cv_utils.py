#!/usr/bin/env python3

"""Utility functions for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -

import pandas
import pytest
from imblearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from hmeasure import h_score
from sklearn.metrics import accuracy_score, f1_score, make_scorer

from ..cv_utils import repeated_nested_cv

# from imblearn.over_sampling import SMOTE


def get_test1_settings():
    pipeline_steps = {'preprocessor': ['skip'],
                      'classifier': ['svm', 'rf']}
    all_params_grid = {
        'skip': {
            'pipe_step_instance': None
        },
        'svm': {
            'pipe_step_instance': SVC(probability=True),
            'C': [1, 10],
            'gamma': [.01, .1],
            'kernel': ['rbf']
        },
        'rf': {
            'pipe_step_instance': RandomForestClassifier(),
            'n_estimators': [1, 10, 15],
            'max_features': [1, 5, 10]
        }
    }
    scorer_selection = [{'score_name': 'H-Measure', 'score_key': 'rank_test_H-Measure',
                         'score_criteria': 'min', 'scorer': make_scorer(h_score, needs_proba=True, pos_label=0)},
                        {'score_name': 'Accuracy', 'score_key': 'rank_test_Accuracy',
                         'score_criteria': 'min', 'scorer': make_scorer(accuracy_score)},
                        {'score_name': 'F1-Score', 'score_key': 'rank_test_F1-Score',
                         'score_criteria': 'min', 'scorer': make_scorer(f1_score)}]

    pipe = Pipeline([('preprocessor', None), ('classifier', None)])
    X, y = load_breast_cancer(return_X_y=True)

    kwargs = {
        'X': X,
        'y': y,
        'param_grid': all_params_grid,
        'steps': pipeline_steps,
        'pipe': pipe,
        'score_selection': scorer_selection
    }
    return kwargs


def test_repeated_nested_cv():
    expected_columns = ['data_name', 'estimator', 'n_repeat', 'outer_fold_n', 'params',
                        'score_name', 'score_value', 'scorer', 'type_classifier',
                        'type_preprocessor']
    # two classifiers, two repeats, two inner folds and three scores
    expected_min_rows = 2 * 2 * 2 * 3

    input_settings = get_test1_settings()

    result = repeated_nested_cv(
        data_name="breast_cancer",
        **input_settings,
        n_repeats=2,
        k_inner_folds=5,
        k_outer_folds=2,
        inner_cv_n_jobs=1,
        verbose_in_cv=1,
        verbose_out_cv=1
    )
    df_out = pandas.DataFrame(result)
    assert (df_out.columns == expected_columns).all()
    assert df_out.shape[0] >= expected_min_rows
