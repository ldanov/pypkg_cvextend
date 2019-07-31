"""Utility functions for running non- and nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -


import copy

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from .eval_grid import EvaluationGrid
from .score_grid import ScoreGrid

# def repeat_cv(data_name: str, X, y, param_dict, steps, pipe,
#               scorer_dict, n_repeats: int = 10, k_folds: int = 5,
#               cv_n_jobs: int = 1, verbose_cv: int = 2):
#     all_scores = []
#     for i in range(n_repeats):
#         grid = get_grid(estimator=pipe,
#                         param_grid=param_grid,
#                         scoring=scorer_dict,
#                         n_splits=k_folds,
#                         random_state=i,
#                         verbose=verbose_cv)
#         grid.fit(X, y)
#         run_score = grid.cv_results_
#         run_score = process_grid_result(run_score,
#                                         step_names,
#                                         data_name=data_name)
#     return all_scores


# from .base import get_grid, get_cv_grid
# from .param_grid import generate_param_grid
# param_grid, step_names = generate_param_grid(steps=steps, param_grid=param_grid)
# cv_grid = get_cv_grid(estimator=pipe,
#                         param_grid=param_grid,
#                         scoring=score_selection.get_sklearn_dict,
#                         cv = StratifiedKFold(shuffle=True, n_splits=5),
#                         verbose=verbose_cv)
# random_states = [0,1]

def nonnested_cv(cv_grid, X, y,
                 step_names,
                 additional_info={'data_name': 'noname'},
                 return_grid=False):
    # TODO: documentation

    if not isinstance(cv_grid, BaseSearchCV):
        raise TypeError('Arg cv_grid must be of class sklearn.model_selection CV types')
    if not isinstance(cv_grid.estimator, Pipeline):
        raise TypeError('Arg estimator of cv_grid must be of class Pipeline')
    
    grid = copy.deepcopy(cv_grid)
    grid.fit(X, y)
    run_score = EvaluationGrid.process_result(grid.cv_results_,
                                              step_names, **additional_info)
    if return_grid:
        return run_score, grid
    return run_score


def nested_cv(cv_grid, X, y,
              step_names: list,
              random_states: list,
              outer_cv=StratifiedKFold(n_splits=2,
                                       shuffle=True,
                                       random_state=1),
              score_selection=ScoreGrid(),
              additional_info={'data_name': 'noname'}):
    # TODO: documentation

    final_result_collector = []
    inner_result_collector = []

    if len(random_states) != outer_cv.n_splits:
        ValueError('Length of random_states arg must equal outer_cv splits.')

    outer_fold = 0
    for indices, random_state in zip(outer_cv.split(X, y), random_states):
        train_index, test_index = indices
        X_train, y_train = X[train_index], y[train_index]
        X_test,  y_test = X[test_index],  y[test_index]

        grid_inner = copy.deepcopy(cv_grid)
        grid_inner.cv.random_state = random_state

        add_info_copy = copy.deepcopy(additional_info)
        add_info_copy['outer_fold_n'] = outer_fold
        add_info_copy['inner_cv_random_state'] = random_state

        inner_run_score, grid_inner = nonnested_cv(grid_inner,
                                                   X_train, y_train,
                                                   step_names=step_names,
                                                   additional_info=add_info_copy,
                                                   return_grid=True)

        grid_evaluate = EvaluationGrid(grid_inner,
                                       score_selection,
                                       step_names)

        outer_score = grid_evaluate.refit_score(X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_test,
                                                y_test=y_test,
                                                **add_info_copy)

        final_result_collector = final_result_collector + outer_score
        inner_result_collector = inner_result_collector + [inner_run_score]

        outer_fold += 1

    return final_result_collector, inner_result_collector
