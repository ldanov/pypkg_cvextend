#!/usr/bin/env python3

"""Utility functions for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -


from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from hmeasure import h_score

from .base import _expand_param_grid, get_grid, process_grid_result
from .grid_search import NestedEvaluationGrid
from .score_grid import ScoreGrid

# TODO: score_selection as a Class
_default_score_selection = [{'score_name': 'H-Measure', 'score_search': 'rank_test_H-Measure',
                             'selector': 'min', 'scorer': make_scorer(h_score, needs_proba=True, pos_label=0)},
                            {'score_name': 'Accuracy', 'score_search': 'rank_test_Accuracy',
                             'selector': 'min', 'scorer': make_scorer(accuracy_score)},
                            {'score_name': 'F1-Score', 'score_search': 'rank_test_F1-Score',
                             'selector': 'min', 'scorer': make_scorer(f1_score)}]


def repeat_cv(data_name: str, X, y, param_grid, steps, pipe,
              scorer_dict, cv_rand_states: list = [], k_folds: int = 5,
              cv_n_jobs: int = 1, verbose_cv: int = 2):

    step_names = list(steps.keys())
    p_grid_exp = _expand_param_grid(steps=steps,
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
                       inner_cv_n_jobs: int = 1, verbose_out_cv: int = 2,
                       verbose_in_cv: int = 2):

    result_collector = []
    p_grid_exp = _expand_param_grid(steps=steps, param_grid=param_grid)
    step_names = list(steps.keys())
    scgrid = ScoreGrid(score_selection)
    scorer_dict = scgrid.get_sklearn_dict()
    # TODO: expose inner cv random_states
    # TODO: expose outer cv random_states
    # TODO: store also inner cv results
    # TODO: intermediary saving of results and restart from point
    for i in range(n_repeats):

        outer_cv = StratifiedKFold(n_splits=k_outer_folds,
                                   shuffle=True,
                                   random_state=i)
        outer_fold = 0
        for train_index, test_index in outer_cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            additional_info = {
                'n_repeat': i,
                'outer_fold_n': outer_fold,
                'data_name': data_name
            }

            grid_inner = get_grid(estimator=pipe,
                                  param_grid=p_grid_exp,
                                  scoring=scorer_dict,
                                  n_splits=k_inner_folds,
                                  random_state=i,
                                  refit=False,
                                  verbose=verbose_in_cv)

            grid_inner.fit(X_train, y_train)

            grid_evaluate = NestedEvaluationGrid(grid_inner,
                                                 scgrid, step_names)

            outer_score = grid_evaluate.refit_score(X_train=X_train,
                                                    y_train=y_train,
                                                    X_test=X_test,
                                                    y_test=y_test,
                                                    **additional_info)

            result_collector = result_collector + outer_score

            outer_fold += 1

    return result_collector
