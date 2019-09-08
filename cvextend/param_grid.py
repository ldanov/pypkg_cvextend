"""Utility function for generating parameter grid
"""

# Authors: Lyubomir Danov <->
# License: -

import itertools

from sklearn.model_selection import ParameterGrid


# TODO: convert to class
# based on https://stackoverflow.com/a/42271829/10960229
def generate_param_grid(steps: dict, param_dict: dict):
    """Generates sklearn.pipeline-compatible param_grid by permutation

    Parameters
    ----------
    steps : dict
        A dictionary of dictionaries. Keys are pipeline steps. Values
        are dicts where key is the relevant key from param_dict and
        value is an instance of the model/callable.
    param_dict : dict
        A dictionary of dictionaries. Keys are str names of models.
        Values are dicts that contain model params as keys and lists of
        values to permute over as values.

    Returns
    -------
    final_params : list of dicts
        Each dict is a permutation of each step's possible values
    step_names : list of str
        The key values of steps

    Examples
    --------
    >>> from cvextend import generate_param_grid
    >>> from sklearn.svm import SVC
    >>> from sklearn.ensemble import RandomForestClassifier
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
    >>> print(generate_param_grid(steps, param_dict))
    """

    final_params = []
    step_value_names = [x.keys() for x in steps.values()]
    for estimator_names in itertools.product(*step_value_names):
        current_grid = {}

        # Step_name and estimator_name should correspond
        # i.e preprocessor must be from pca and select.
        for step_name, estimator_name in zip(steps.keys(), estimator_names):

            # Grab
            current_grid[step_name] = [steps[step_name][estimator_name]]

            for param, value in param_dict.get(estimator_name).items():
                # Set parameters corresponding to above estimator
                current_grid[step_name + '__' + param] = value
        # Append this dictionary to final params
        final_params.append(current_grid)

    try:
        ParameterGrid(final_params)
    except Exception as e:
        raise e
    step_names = list(steps.keys())
    return final_params, step_names
