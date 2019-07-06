#!/usr/bin/env python3

"""Utility functions for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -


from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .base import generate_param_grid, get_grid, process_grid_result
from .grid_search import NestedEvaluationGrid
from .score_grid import ScoreGrid




def repeat_cv(data_name: str, X, y, param_grid, steps, pipe,
              scorer_dict, cv_rand_states: list = [], k_folds: int = 5,
              cv_n_jobs: int = 1, verbose_cv: int = 2):

    step_names = list(steps.keys())
    p_grid_exp = generate_param_grid(steps=steps,
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

# TODO: outer cv given as instantiated object
# TODO: inner cv grid given as instantiated object
def repeated_nested_cv(data_name: str, X, y, param_grid, steps, pipe,
                       n_repeats, k_inner_folds=5, k_outer_folds=2,
                       score_selection=ScoreGrid(),
                       inner_cv_n_jobs: int = 1, verbose_out_cv: int = 2,
                       verbose_in_cv: int = 2):

    result_collector = []
    p_grid_exp = generate_param_grid(steps=steps, param_grid=param_grid)
    step_names = list(steps.keys())

    if not isinstance(score_selection, ScoreGrid):
        TypeError('score_selection is not a ScoreGrid instance')
    scorer_dict = score_selection.get_sklearn_dict()

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
                                                 score_selection,
                                                 step_names)

            outer_score = grid_evaluate.refit_score(X_train=X_train,
                                                    y_train=y_train,
                                                    X_test=X_test,
                                                    y_test=y_test,
                                                    **additional_info)

            result_collector = result_collector + outer_score

            outer_fold += 1

    return result_collector
