"""Utility functions for running non- and nested cross-validation of sampling methods
"""

# Authors: Lyubomir Danov <->
# License: -


import copy

from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .eval_grid import EvaluationGrid
from .score_grid import ScoreGrid


def basic_cv(cv_grid, X, y, additional_info=None):
    """Run basic cross-validation.

    Parameters
    ----------
    cv_grid : object
        An instance inheriting from `sklearn.BaseSearchCV`. Its estimator
        has to inherit from `sklearn.Pipeline`.
    X : array-like
        Array of data to be used for training and validation
    y : array-like
        Target relative to X
    additional_info : dict
        Any additional information to be inserted in the cv results.

    Returns
    -------
    run_score : dict-like
        The `grid.cv_results_` enchanced with additional_info
    grid : object
        The fitted grid object

    Raises
    ------
    TypeError
        if cv_grid does not inherit from `sklearn.BaseSearchCV` or
        if cv_grid.estimator does not inherit from `sklearn.Pipeline`

    Examples
    --------
    >>> from cvextend import basic_cv
    >>> from cvextend import generate_param_grid
    >>> from cvextend import ScoreGrid
    >>> import pandas
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.svm import SVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import GridSearchCV, StratifiedKFold
    >>> from sklearn.pipeline import Pipeline
    >>> scorer_selection = ScoreGrid(scorers)
    >>> sk_score = scorer_selection.get_sklearn_dict()
    >>> pipe = Pipeline([('preprocessor', None), ('classifier', None)])
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> steps = {
    ...     'preprocessor': {'skip': None},
    ...     'classifier': {
    ...         'svm': SVC(probability=True),
    ...         'rf': RandomForestClassifier()
    ...     }
    ... }
    >>> param_dict = {
    ...     'skip': {},
    ...     'svm': {'C': [1, 10, 100],
    ...             'gamma': [.01, .1],
    ...             'kernel': ['rbf']},
    ...     'rf': {'n_estimators': [1, 10, 100],
    ...         'max_features': [1, 5, 10, 20]}
    ... }
    >>> params, steps = generate_param_grid(steps=steps,
    ...                                     param_dict=param_dict)
    >>> inner_cv_use = StratifiedKFold(n_splits=5, shuffle=True,
    ...                                random_state=0)
    >>> test_cv_grid = GridSearchCV(estimator=pipe,
    ...                             param_grid=params,
    ...                             scoring=sk_score,
    ...                             cv=inner_cv_use,
    ...                             refit=False)
    >>> result_basic = basic_cv(test_cv_grid, X, y, )

    """

    if not isinstance(cv_grid, BaseSearchCV):
        raise TypeError('cv_grid must inherit from sklearn BaseSearchCV')
    if not isinstance(cv_grid.estimator, Pipeline):
        raise TypeError('cv_grid.estimator must inherit from sklearn Pipeline')

    if additional_info is None:
        additional_info = {}

    step_names = list(cv_grid.estimator.named_steps.keys())
    grid = copy.deepcopy(cv_grid)
    grid.fit(X, y)

    run_score, _ = EvaluationGrid.process_result(grid.cv_results_,
                                                 step_names)

    run_score = EvaluationGrid.add_info(run_score, **additional_info)

    return run_score, grid


def nested_cv(cv_grid, X, y, inner_cv_seeds: list,
              outer_cv=StratifiedKFold(n_splits=5, random_state=1),
              score_selection=ScoreGrid(),
              additional_info=None):
    """Run nested cross-validation.

    Parameters
    ----------
    cv_grid : object
        An instance inheriting from `sklearn.BaseSearchCV`. Its estimator
        has to inherit from sklearn Pipeline. Its cv parameter must be
        set in order to be a CV splitter that has random_state attribute
        (https://scikit-learn.org/stable/glossary.html#term-cv-splitter)
    X : array-like
        Array of data to be used for training, validation and testing
    y : array-like
        Target relative to X
    inner_cv_seeds : list
        list of seeds, assigned on outer_cv split. Length of list
    outer_cv : object
        An instance inheriting from `sklearn.BaseCrossValidator`. Used
        for outer cross-validation split of X. Needs to have the n_splits
        attribute.
    score_selection : object
        An instance of ScoreGrid.
    additional_info : dict
        Any additional information to be inserted in the inner and outer
        cv results.

    Raises
    ------
    TypeError
        if outer_cv does not inherit from `sklearn.BaseCrossValidator`
    ValueError
        if the number of splits in outer_cv and length of random_state
        differ

    Returns
    -------
    outer_results : dict
        Contains the performance of the best hyperparameters of each
        estimator (combination of Pipeline steps) on each outer fold.
        The best performing hyperparameter combination is picked on a
        per score basis from the inner (nested) cross-validation. Only
        the score, for which the combination was a winner, is reported.
    inner_results : list of dicts
        Contains all results from nested cross-validation as reported by
        `BaseSearchCV.cv_results_` for each outer split

    Examples
    --------
    >>> from cvextend import nested_cv
    >>> from cvextend import generate_param_grid
    >>> from cvextend import ScoreGrid
    >>> import pandas
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.svm import SVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import GridSearchCV, StratifiedKFold
    >>> from sklearn.pipeline import Pipeline
    >>> steps = {
    ...     'preprocessor': {'skip': None},
    ...     'classifier': {
    ...         'svm': SVC(probability=True),
    ...         'rf': RandomForestClassifier()
    ...     }
    ... }
    >>> param_dict = {
    ...     'skip': {},
    ...     'svm': {'C': [1, 10, 100],
    ...             'gamma': [.01, .1],
    ...             'kernel': ['rbf']},
    ...     'rf': {'n_estimators': [1, 10, 100],
    ...         'max_features': [1, 5, 10, 20]}
    ... }
    >>> scorer_selection = ScoreGrid(scorers)
    >>> sk_score = scorer_selection.get_sklearn_dict()
    >>> pipe = Pipeline([('preprocessor', None), ('classifier', None)])
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> params, steps = generate_param_grid(steps=steps,
    ...                                     param_dict=param_dict)
    >>> inner_cv_use = StratifiedKFold(n_splits=5, shuffle=True,
    ...                                random_state=0)
    >>> inner_cv_seeds = [1,2]
    >>> test_cv_grid = GridSearchCV(estimator=pipe,
    ...                             param_grid=params,
    ...                             scoring=sk_score,
    ...                             cv=inner_cv_use,
    ...                             refit=False)
    >>> outer_cv_use = StratifiedKFold(n_splits=2, random_state=1,
    ...                                shuffle=True)
    >>> addit_info = {'dataset_name': "breast_cancer"}
    >>> result_outer, result_inner = nested_cv(cv_grid=test_cv_grid,
    ...                                        X=X, y=y,
    ...                                        score_selection=scorer_selection,
    ...                                        inner_cv_seeds=inner_cv_seeds,
    ...                                        outer_cv=outer_cv_use,
    ...                                        additional_info=addit_info
    ...                                        )
    >>> print(pandas.DataFrame(result_outer))
    >>> print(pandas.concat([pandas.DataFrame(x) for x in result_inner]))

    """

    outer_results = []
    inner_results = []

    if not (len(inner_cv_seeds) == outer_cv.n_splits or len(inner_cv_seeds) == 1):
        raise ValueError('Length of inner_cv_seeds must equal outer_cv splits')

    if not isinstance(outer_cv, BaseCrossValidator):
        raise TypeError('outer_cv must be of class sklearn BaseCrossValidator')

    if not isinstance(cv_grid.cv, BaseCrossValidator):
        raise TypeError('inner_cv used in cv_grid must be of '
                        'class sklearn BaseCrossValidator')

    if additional_info is None:
        additional_info = {'dataset_name': 'unknown'}

    outer_fold = 0
    for indices, random_state in zip(outer_cv.split(X, y), inner_cv_seeds):
        train_index, test_index = indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid_inner = copy.deepcopy(cv_grid)
        # TODO better random state assignment
        grid_inner.cv.random_state = random_state

        add_info_copy = copy.deepcopy(additional_info)
        add_info_copy['outer_fold_n'] = outer_fold
        add_info_copy['inner_cv_random_state'] = random_state

        inner_score, grid_inner_fitted = basic_cv(grid_inner,
                                                  X_train, y_train,
                                                  add_info_copy)

        grid_evaluate = EvaluationGrid(grid_inner_fitted,
                                       score_selection)

        outer_score = grid_evaluate.refit_score(X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_test,
                                                y_test=y_test,
                                                **add_info_copy)

        outer_results = outer_results + outer_score
        inner_results = inner_results + [inner_score]

        outer_fold += 1

    return outer_results, inner_results
