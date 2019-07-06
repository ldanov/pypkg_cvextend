#!/usr/bin/env python3

"""Tests for param_grid"""

# Authors: Lyubomir Danov <->
# License: -

import pytest

from ..base import generate_param_grid
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def get_test_case():

    pipeline_steps = {'preprocessor': ['skip'],
                      'classifier': ['svm', 'rf']}
    all_params_grid = {
        'skip': {
            'pipe_step_instance': None
        },
        'svm': {
            'pipe_step_instance': SVC(probability=True),
            'C': [1, 10, 100],
            'gamma': [.01, .1],
            'kernel': ['rbf']
        },
        'rf': {
            'pipe_step_instance': RandomForestClassifier(),
            'n_estimators': [1, 10, 100],
            'max_features': [1, 5, 10, 20]
        }
    }

    return pipeline_steps, all_params_grid


def test_generate_param_grid():

    steps, pgrid = get_test_case()

    exp_result = [
        {
            'preprocessor': [None],
            'classifier': [SVC(probability=True)],
            'classifier__C': [1, 10, 100],
            'classifier__gamma': [0.01, 0.1],
            'classifier__kernel': ['rbf']
        },
        {
            'preprocessor': [None],
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [1, 10, 100],
            'classifier__max_features': [1, 5, 10, 20]
        }
    ]

    result = generate_param_grid(steps, pgrid)
    for d1, d2 in zip(result, exp_result):
        for key in d1.keys():
            if key != "classifier":
                assert d1[key] == d2[key]
            else:
                assert d1[key].__class__ == d2[key].__class__