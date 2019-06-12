#!/usr/bin/env python3

"""Utility functions for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -


from .base import add_class_name
import pandas
import numpy
import copy
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from hmeasure import h_score
# from imblearn.pipeline import Pipeline
from .base import _expand_param_grid, _get_object_fullname
from .base import get_grid, process_grid_result, transform_score_selection

_default_score_selection = [{'score_name': 'H-Measure', 'score_search': 'rank_test_H-Measure',
                             'selector': 'min', 'scorer': make_scorer(h_score, needs_proba=True, pos_label=0)},
                             {'score_name': 'Accuracy', 'score_search': 'rank_test_Accuracy',
                             'selector': 'min', 'scorer': make_scorer(accuracy_score)},
                             {'score_name': 'F1-Score', 'score_search': 'rank_test_F1-Score',
                             'selector': 'min', 'scorer': make_scorer(f1_score)}]


def repeat_cv(data_name: str, X, y, param_grid, steps, pipe,
              scorer_dict, cv_rand_states: list = [], k_folds: int = 5,
              cv_n_jobs: int = 1, verbose_cv: int = 2):

    p_grid_exp, step_names = _expand_param_grid(steps=steps,
                                                param_grid=param_grid)

    all_scores = []
    for state in cv_rand_states:

        grid = get_grid(estimator=pipe,
                        param_grid=p_grid_exp,
                        scoring=scorer_dict,
                        n_splits=k_folds,
                        random_state=state,
                        verbose=verbose_cv)

        grid.fit(X, y)
        run_score = grid.cv_results_
        run_score = process_grid_result(run_score, step_names, data_name)
        all_scores.append(run_score)
    return all_scores


def repeated_nested_cv(data_name: str, X, y, param_grid, steps, pipe,
                       n_repeats, k_inner_folds=5, k_outer_folds=2,
                       score_selection=_default_score_selection,
                       inner_cv_n_jobs: int = 1, verbose_out_cv: int = 2, verbose_in_cv: int = 2):

    result_collector = []
    p_grid_exp, step_names = _expand_param_grid(steps=steps,
                                                param_grid=param_grid)

    scorer_dict = transform_score_selection(score_selection)
    # TODO: expose inner and outer cv random_states
    for i in range(n_repeats):

        outer_cv = StratifiedKFold(n_splits=k_outer_folds,
                                   shuffle=True,
                                   random_state=i)
        outer_fold = 0
        for train_index, test_index in outer_cv.split(X, y):
            outer_fold += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            grid_inner = get_grid(estimator=pipe,
                                  param_grid=p_grid_exp,
                                  scoring=scorer_dict,
                                  n_splits=k_inner_folds,
                                  random_state=i,
                                  refit=False,
                                  verbose=verbose_in_cv)

            grid_inner.fit(X_train, y_train)

            selected_inner_winners = get_cv_winners(grid_inner, X_train=X_train, y_train=y_train,
                                                    step_names=step_names,
                                                    score_selection=score_selection)

            added_info = {
                'n_repeat': i,
                'outer_fold_n': outer_fold,
                'data_name': data_name
            }
            outer_score = get_scores(selected_inner_winners,
                                     score_selection,
                                     X_test, y_test, added_info)

            result_collector = result_collector + outer_score

    return result_collector


def get_cv_winners(grid_inner, step_names, score_selection, X_train, y_train):
    '''
    Given a fitted BaseSearchCV object return a list of dictionaries containing
    fitted estimators for each score in the BaseSearchCV object
    '''

    evaluation_list = grid_inner.cv_results_

    scorers_best_params = get_best_params(eval_list=evaluation_list,
                                          score_selection=score_selection,
                                          step_names=step_names)

    for best_param in scorers_best_params:
        cloned_estim = copy.deepcopy(grid_inner.estimator)
        cloned_estim.set_params(**best_param['params'])
        cloned_estim.fit(X_train, y_train)
        best_param['estimator'] = cloned_estim

    return scorers_best_params


def get_best_params(eval_list, score_selection, step_names):
    eval_df = copy.deepcopy(pandas.DataFrame(eval_list))
    eval_df, types_all_steps = add_class_name(eval_df, step_names)
    per_score = []

    for score_type in score_selection:

        score_key = score_type['score_search']
        selector = score_type['selector']

        # which columns to select
        retr_cols = types_all_steps + ['params']
        # for each unique value in each step from step_names
        # return those entries, where score_key is selector
        idx = eval_df.groupby(types_all_steps)[score_key].transform(selector)
        score_best_params = copy.deepcopy(
            eval_df.loc[idx == eval_df[score_key], retr_cols])

        # return score_name and scorer itself for ease of scoring
        score_best_params['score_name'] = score_type['score_name']
        score_best_params['scorer'] = score_type['scorer']

        per_score = per_score + score_best_params.to_dict('records')

    return per_score


def get_scores(selected_inner_winners, score_selection, X_test, y_test, added_info):
    # candidate_list
    for estimator_dict in selected_inner_winners:
        scorer = estimator_dict['scorer']
        estimator = estimator_dict['estimator']
        result = scorer(estimator, X_test, y_test)
        estimator_dict['score_value'] = result
        for key, value in added_info.items():
            estimator_dict[key] = value
    return selected_inner_winners
