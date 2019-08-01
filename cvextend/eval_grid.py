"""EvaluationGrid class for running nested cross-validation of sampling methods
"""

# Authors: Lyubomir Danov <->
# License: -

import copy

import pandas

from .score_grid import ScoreGrid


class EvaluationGrid:
    """A class that given a fitted `sklearn.BaseSearchCV` object returns
    the best estimator's performance on a separate test set for each score. 
    Requires original X and y used for training.

    Parameters
    ----------
    gridcv : object
        A fitted `sklearn.model_selection.GridSearchCV` or 
        `sklearn.model_selection.RandomizedSearchCV` instance

    score_grid : object
        A `cvextend.ScoreGrid` instance

    """

    def __init__(self, gridcv, score_grid):

        if not isinstance(score_grid, ScoreGrid):
            TypeError("score_grid does not inherit from ScoreGrid!")

        group_names = list(gridcv.estimator.named_steps.keys())
        eval_df = pandas.DataFrame(copy.deepcopy(gridcv.cv_results_))
        eval_df, group_type_keys = self.process_result(eval_df, group_names)

        self.eval_df = eval_df
        self.group_type_keys = group_type_keys

        self.cv_estimator = copy.deepcopy(gridcv.estimator)
        self.score_grid = score_grid

        # Instantiate for state tracking
        self.scorers_best_params = None
        self.fitted_estimators = None
        self.final_result = None

    def refit_score(self, X_train, y_train, X_test, y_test, **info):
        """Finds the best hyperparameters for each estimator, refits
        them on X_train and y_train and reports the performance on X_test
        and y_test

        Parameters
        ----------
        X_train : array-like
            Array of training data
        y_train : array-like
            Target relative to X_train
        X_test : array-like
            Array of testing data
        y_test : array-like
            Target relative to X_test
        **info : dict of str
            Info to be added to final score object (e.g. dataset name)

        """
        self.get_best_params()
        self.get_fitted_estimators(X_train, y_train)
        self.get_scores(X_test, y_test, **info)

        return self.final_result

    def get_best_params(self):
        """Given a `BaseSearchCV.cv_results_` object, find the estimator 
        hyperparameters that had the best performance for each of the 
        score_grid scores. An estimator is a combination of each Pipeline 
        step type.

        """
        # TODO: replace pandas with numpy
        eval_df = self.eval_df

        # which columns to select
        params = [col for col in eval_df if col.startswith('param_')]
        retr_cols = self.group_type_keys + params + ['params']

        per_score = []

        for score_type in self.score_grid.score_selection:

            score_key = score_type['score_key']
            score_criteria = score_type['score_criteria']

            # for each unique value in each group from groups
            # return entries where score_key corresponds to score_criteria
            idx = eval_df.groupby(self.group_type_keys)[score_key]
            idx = idx.transform(score_criteria)

            tmp_df = eval_df.loc[eval_df[score_key] == idx, retr_cols]
            score_best_params = copy.deepcopy(tmp_df)

            # return score_name and scorer itself for ease of scoring
            score_best_params['score_name'] = score_type['score_name']
            score_best_params['scorer'] = score_type['scorer']

            per_score = per_score + score_best_params.to_dict('records')

        self.scorers_best_params = per_score
        return self

    def get_fitted_estimators(self, X_train, y_train):
        """Given an the best estimator hyperparameters explicitly refit
        each estimator. Is used when refitting after nested cross-
        validation.

        Parameters
        ----------
        X_train : array-like
            Array of training data
        y_train : array-like
            Target relative to X_train

        """
        if self.scorers_best_params is None:
            ValueError('self.get_best_params() has not been run')

        scorers_best_params = copy.deepcopy(self.scorers_best_params)
        for best_param in scorers_best_params:
            cloned_estim = copy.deepcopy(self.cv_estimator)
            cloned_estim.set_params(**best_param['params'])
            cloned_estim.fit(X_train, y_train)
            best_param['estimator'] = cloned_estim

        self.fitted_estimators = scorers_best_params
        return self

    def get_scores(self, X_test, y_test, **info):
        """Given a `BaseSearchCV.cv_results_` object with results of all 
        parameter combinations, return a list of dictionaries containing 
        the best hyperparameters for each combination of score and 
        Pipeline step. 

        Parameters
        ----------
        X_test : array-like
            Array of testing data
        y_test : array-like
            Target relative to X_test
        **info : dict of str
            Info to be added to final score object (e.g. dataset name)

        """
        # candidate_list

        if self.fitted_estimators is None:
            ValueError('self.get_fitted_estimators(X, y) has not been run')

        final_result = copy.deepcopy(self.fitted_estimators)
        for estimator_dict in final_result:

            estimator = estimator_dict['estimator']
            scorer = estimator_dict['scorer']
            result = scorer(estimator, X_test, y_test)

            estimator_dict['score_value'] = result

            estimator_dict = EvaluationGrid.add_info(estimator_dict, **info)

        self.final_result = final_result
        return self

    @staticmethod
    def add_info(data, **info):
        """Add information to a dict-like

        Parameters
        ----------
        data : dict-like
            Object to add information to
        **info : dict
            Key-value pairs to be added

        """
        for key, value in info.items():
            data[key] = value
        return data

    @staticmethod
    def get_object_fullname(obj):
        """Given an object, return a string of module.class.name

        Parameters
        ----------
        obj : object

        Returns
        -------
        fin_str : str
        """

        module = obj.__class__.__module__

        if module is None or module == str.__class__.__module__:
            fin_str = obj.__class__.__name__
        else:
            fin_str = module + '.' + obj.__class__.__name__

        return fin_str

    @staticmethod
    def process_result(result, step_names):
        """Given original results dict or df as given by `BaseSearchCV` and 
        pipeline step names, enchances the results with the type of each 
        transformer or estimator from each step of the pipeline.

        Parameters
        ----------
        result : dict
            The original results of `BaseSearchCV.cv_results_`
        step_names : list
            The str names of the pipeline steps of the estimator given 
            to `BaseSearchCV` for fitting.

        Returns
        -------
        result : dict
            Enchanced result object
        group_type_keys : list
            List of keys of newly added entries

        """

        # due to specifying steps in Pipeline as object instances,
        # results contain the instances themselves
        # instead return class name as string
        obj_fullname = EvaluationGrid.get_object_fullname

        group_type_keys = []
        for group in step_names:
            type_group = 'type_' + group
            param_group = 'param_' + group

            classes = result[param_group]
            result[type_group] = [obj_fullname(x) for x in classes]

            group_type_keys.append(type_group)

        return result, group_type_keys
