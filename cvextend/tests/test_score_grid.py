"""Tests for ScoreGrid"""

# Authors: Lyubomir Danov <->
# License: -

import copy

import pytest
from sklearn.metrics import accuracy_score, f1_score, make_scorer

from hmeasure import h_score

from ..score_grid import ScoreGrid


def get_correct_data():
    sc_selection = [
        {
            'score_name': 'H-Measure', 
            'score_key': 'rank_test_H-Measure',
            'score_criteria': 'min', 
            'scorer': make_scorer(h_score, needs_proba=True, pos_label=0)
        },
        {
            'score_name': 'Accuracy', 
            'score_key': 'rank_test_Accuracy',
            'score_criteria': 'min', 
            'scorer': make_scorer(accuracy_score)
        },
        {
            'score_name': 'F1-Score', 
            'score_key': 'rank_test_F1-Score',
            'score_criteria': 'min', 
            'scorer': make_scorer(f1_score)
        }
    ]
    exp_sklearn = {
        'H-Measure': make_scorer(h_score, needs_proba=True, pos_label=0),
        'Accuracy': make_scorer(accuracy_score),
        'F1-Score': make_scorer(f1_score)
    }

    return sc_selection, exp_sklearn


_cases_to_run = [
    "case1",
    "case2",
    "case3"
]


@pytest.fixture(params=_cases_to_run)
def get_wrong_input_data(request):
    _default_case = [{
        'score_name': 'H-Measure', 
        'score_key': 'rank_test_H-Measure',
        'score_criteria': 'min', 
        'scorer': make_scorer(h_score, needs_proba=True, pos_label=0)
    }]

    case1 = copy.deepcopy(_default_case)
    del case1[0]['score_name']

    case2 = copy.deepcopy(_default_case)
    case2[0]['scorer'] = str(case2[0]['scorer'])

    case3 = copy.deepcopy(_default_case)
    case3[0]['score_criteria'] = min

    sc_selection = {
        'case1': case1,
        'case2': case2,
        'case3': case3
    }
    exp_error = {
        'case1': KeyError,
        'case2': TypeError,
        'case3': TypeError
    }
    return sc_selection[request.param], exp_error[request.param]


def test_score_grid_correct():
    correct_input, correct_out_sklearn = get_correct_data()

    sc_grid = ScoreGrid(correct_input)
    out_sklearn = sc_grid.get_sklearn_dict()

    assert correct_out_sklearn.keys() == out_sklearn.keys()
    for key, value in out_sklearn.items():
        assert isinstance(value, correct_out_sklearn[key].__class__)


def test_score_grid_wrong_input(get_wrong_input_data):
    with pytest.raises(get_wrong_input_data[1]):
        ScoreGrid(get_wrong_input_data[0])
