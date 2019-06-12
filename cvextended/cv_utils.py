#!/usr/bin/env python3

"""Utility functions for running nested cross-validation of sampling methods"""

# Authors: Lyubomir Danov <->
# License: -


import numpy
import copy
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from imblearn.pipeline import Pipeline
from .param_grid import generate_param_grids


def _expand_param_grid(steps, param_grid):
    param_grid_expanded = generate_param_grids(steps, param_grid)
    step_names = list(steps.keys())

    try:
        _ = ParameterGrid(param_grid_expanded)
    except Exception as e:
        raise e

    return param_grid_expanded, step_names


def _get_object_fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    else:
        return module + '.' + o.__class__.__name__


def get_grid(estimator, param_grid, scoring, n_splits, random_state, verbose):
    sfk_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True,
        random_state=random_state)

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=sfk_cv,
        scoring=scoring,
        return_train_score=True,
        iid=False,
        refit='H-Measure',
        verbose=verbose
    )
    return grid


def process_grid_result(grid_result, step_names, data_name):
    grid_res = copy.deepcopy(grid_result)
    grid_res['data_name'] = data_name

    full_step_names = ['param_' + str(x) for x in step_names]

    # due to specifying steps in Pipeline as object instances,
    # results contain the instances themselves
    # instead return class name as string
    for step in full_step_names:
        object_class = []
        for obj in grid_res[step]:
            object_class.append(_get_object_fullname(obj))
        object_class = numpy.array(object_class)
        grid_res[step] = object_class

    return grid_res


def get_model_scores(data_name: str, X, y,
                     param_grid, step_names, pipe,
                     k_folds, scorer_dict, cv_rand_state=0,
                     memory=None, cv_n_jobs: int = 1,
                     verbose_cv: int = 2):

    grid = get_grid(estimator=pipe,
                    param_grid=param_grid,
                    scoring=scorer_dict,
                    n_splits=k_folds,
                    random_state=cv_rand_state,
                    verbose=verbose_cv)

    grid.fit(X, y)
    grid_result_df = process_grid_result(grid.cv_results_,
                                         step_names=step_names,
                                         data_name=data_name)

    return grid_result_df


def repeat_cv(data_name: str, X, y, steps, param_grid, pipe,
                              k_folds, scorer_dict, cv_rand_states: list = [],
                              memory: bool = None, cv_n_jobs: int = 1,
                              verbose_cv: int = 2):

    p_grid_exp, step_names = _expand_param_grid(steps=steps,
                                                param_grid=param_grid)

    all_scores = []
    for state in cv_rand_states:
        run_score = get_model_scores(cv_rand_state=state,
                                     data_name=data_name,
                                     X=X, y=y, step_names=step_names,
                                     param_grid=p_grid_exp, pipe=pipe,
                                     k_folds=k_folds, scorer_dict=scorer_dict,
                                     memory=memory, cv_n_jobs=cv_n_jobs,
                                     verbose_cv=verbose_cv)
        all_scores.append(run_score)
    return all_scores


def repeated_nested_cv(data_name: str, X, y, param_grid, steps, pipe,
              scorer_dict, n_repeats, k_inner_folds=5, k_outer_folds=2,
              memory=None, inner_cv_n_jobs: int = 1,
              verbose_out_cv: int = 2,
              verbose_in_cv: int = 2):

    result_collector = []
    p_grid_exp, step_names = _expand_param_grid(steps=steps,
                                                param_grid=param_grid)

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

            inner_score = get_model_scores(data_name=data_name,
                                           X=X_train, y=y_train,
                                           step_names=step_names, pipe=pipe,
                                           scorer_dict=scorer_dict,
                                           param_grid=p_grid_exp,
                                           k_folds=k_inner_folds,
                                           cv_rand_state=i,
                                           memory=memory,
                                           cv_n_jobs=inner_cv_n_jobs,
                                           verbose_cv=verbose_in_cv)

            estimate_on = get_inner_cv_winners(inner_score)
            outer_score = get_outer_scores_inner_winners(estimate_on,
                                                         scorer_dict,
                                                         X_test, y_test)

            outer_score['n_repeat'] = i
            outer_score['outer_fold'] = outer_fold
            outer_score['data_name'] = data_name

            result_collector.append(outer_score)

    return result_collector


def get_inner_cv_winners(inner_score):
    raise NotImplementedError


def get_outer_scores_inner_winners(estimate_on, scorer_dict, X_test, y_test):
    raise NotImplementedError
