#!/usr/bin/env python3

"""Utility function for generating parameter grid"""

# Authors: Lyubomir Danov <->
# License: -

from itertools import product as iter_product

from sklearn.model_selection import ParameterGrid


# TODO: convert to class
def generate_param_grid(steps: dict, param_dict):
    '''
    Generates sklearn.pipeline-compatible param_grid by permutation

    steps: dict
        Contains the keys to each pipeline step and which param_dict keys to permute over

    param_dict: dict
        Keys are str names of models/callables
        Values are dicts that must contain the key 'pipe_step_instance', with value
        model/callable instance. All other keys are model params and values are
        lists of values to permute over.

    pipeline_steps = {'preprocessor': ['skip'],
                      'classifier': ['svm', 'rf']}
    all_params_grid = {
        'skip': {
            'pipe_step_instance': None
        },
        'svm': {
            'pipe_step_instance': SVC(probability=True),
            'C': [1, 10],
            'gamma': [.01, .1],
            'kernel': ['rbf']
        },
        'rf': {
            'pipe_step_instance': RandomForestClassifier(),
            'n_estimators': [1, 10, 15],
            'max_features': [1, 5, 10]
        }
    }
    '''
    # TODO: refactor s.t. steps contain the instances?

    final_params = []

    for estimator_names in iter_product(*steps.values()):
        current_grid = {}

        # Step_name and estimator_name should correspond
        # i.e preprocessor must be from pca and select.
        for step_name, estimator_name in zip(steps.keys(), estimator_names):
            for param, value in param_dict.get(estimator_name).items():
                if param == 'pipe_step_instance':
                    # Set actual estimator in pipeline
                    current_grid[step_name] = [value]
                else:
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


# # import from https://stackoverflow.com/a/42271829/10960229
# def generate_param_grids(steps, param_grids):

#     final_params = []
#     # step_keys, step_values = steps.items()

#     for estimator_names in iter_product(*steps.values()):
#         current_grid = {}

#         # Step_name and estimator_name should correspond
#         # i.e preprocessor must be from pca and select.
#         for step_name, estimator_name in zip(steps.keys(), estimator_names):
#             for param, value in param_grids.get(estimator_name).items():
#                 if param == 'pipe_step_instance':
#                     # Set actual estimator in pipeline
#                     current_grid[step_name] = [value]
#                 else:
#                     # Set parameters corresponding to above estimator
#                     current_grid[step_name + '__' + param] = value
#         # Append this dictionary to final params
#         final_params.append(current_grid)

#     return final_params

# add all the estimators you want to "OR" in single key
# use OR between `pca` and `select`,
# use OR between `svm` and `rf`
# different keys will be evaluated as serial estimator in pipeline
# pipeline_steps = {'preprocessor':['pca', 'select'],
#                   'classifier':['svm', 'rf']}

# # fill parameters to be searched in this dict
# all_param_grids = {'svm':{'pipe_step_instance':SVC(),
#                           'C':[0.1,0.2]
#                          },

#                    'rf':{'pipe_step_instance':RandomForestClassifier(),
#                          'n_estimators':[10,20]
#                         },

#                    'pca':{'pipe_step_instance':PCA(),
#                           'n_components':[10,20]
#                          },

#                    'select':{'pipe_step_instance':SelectKBest(),
#                              'k':[5,10]
#                             }
#                   }
# Call the method on the above declared variables
# param_grids_list = make_param_grids(pipeline_steps, all_param_grids)
