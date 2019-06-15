#!/usr/bin/env python3

"""Tests for ScoreGrid"""

# Authors: Lyubomir Danov <->
# License: -

import pytest
from ..score_grid import ScoreGrid
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from hmeasure import h_score
import copy


def get_correct_data():
    sc_selection = [
        {
            'score_name': 'H-Measure', 'score_criterion_name': 'rank_test_H-Measure',
            'score_criterion_selector': 'min', 'scorer': make_scorer(h_score, needs_proba=True, pos_label=0),
            'use_for_selection': True
        },
        {
            'score_name': 'Accuracy', 'score_criterion_name': 'rank_test_Accuracy',
            'score_criterion_selector': 'min', 'scorer': make_scorer(accuracy_score),
            'use_for_selection': False
        },
        {
            'score_name': 'F1-Score', 'score_criterion_name': 'rank_test_F1-Score',
            'score_criterion_selector': 'min', 'scorer': make_scorer(f1_score),
            'use_for_selection': True
        }
    ]
    exp_sklearn = {
        'H-Measure': make_scorer(h_score, needs_proba=True, pos_label=0),
        'Accuracy': make_scorer(accuracy_score),
        'F1-Score': make_scorer(f1_score)
    }
    exp_selection = [sc_selection[0], sc_selection[2]]
    return sc_selection, exp_sklearn, exp_selection


_cases_to_run = [
    "case1",
    "case2",
    "case3"
]


@pytest.fixture(params=_cases_to_run)
def get_wrong_input_data(request):
    _default_case = [{
        'score_name': 'H-Measure', 'score_criterion_name': 'rank_test_H-Measure',
        'score_criterion_selector': 'min', 'scorer': make_scorer(h_score, needs_proba=True, pos_label=0),
        'use_for_selection': True
    }]

    case1 = copy.deepcopy(_default_case)
    del case1[0]['score_name']

    case2 = copy.deepcopy(_default_case)
    case2[0]['scorer'] = str(case2[0]['scorer'])

    case3 = copy.deepcopy(_default_case)
    case3[0]['use_for_selection'] = str(case2[0]['use_for_selection'])

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
    correct_input, correct_out_sklearn, correct_out_select = get_correct_data()
    print(correct_out_select)

    sc_grid = ScoreGrid(correct_input)
    out_sklearn = sc_grid.get_sklearn_dict()
    out_select = sc_grid.get_selection_scores()
    print(out_select)

    assert correct_out_sklearn.keys() == out_sklearn.keys()
    for key, value in out_sklearn.items():
        assert isinstance(value, correct_out_sklearn[key].__class__)
    for returned_dict, correct_dict in zip(out_select, correct_out_select):
        assert returned_dict.keys() == correct_dict.keys()
        for key, value in returned_dict.items():
            assert isinstance(value, correct_dict[key].__class__)


def test_score_grid_wrong_input(get_wrong_input_data):
    with pytest.raises(get_wrong_input_data[1]):
        ScoreGrid(get_wrong_input_data[0])
