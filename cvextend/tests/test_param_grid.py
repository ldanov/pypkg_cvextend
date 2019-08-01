"""Tests for param_grid
"""

# Authors: Lyubomir Danov <->
# License: -

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ..param_grid import generate_param_grid


def get_test_case():

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

    return pipeline_steps, params_dict


_exp_grid_out = [
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


def test_generate_param_grid():

    steps, pdict = get_test_case()
    exp_params = _exp_grid_out
    exp_names = list(steps.keys())

    params, stepnames = generate_param_grid(steps, pdict)

    assert len(params) == 2
    assert len(stepnames) == len(exp_names)
    assert isinstance(params[0], dict)
    assert isinstance(params[1], dict)

    assert stepnames == exp_names

    for d1, d2 in zip(params, exp_params):
        for key in d1.keys():
            if key != "classifier":
                assert d1[key] == d2[key]
            else:
                assert d1[key].__class__ == d2[key].__class__

    for key, value in pdict['svm'].items():
        if key == 'pipe_step_instance':
            continue
        else:
            assert value == params[0]['classifier__' + key]
