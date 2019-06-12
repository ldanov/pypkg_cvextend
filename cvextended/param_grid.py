#!/usr/bin/env python3

"""Utility functions for generating parameter grid"""

# Authors: Lyubomir Danov <->
# License: -

import itertools

# import from https://stackoverflow.com/a/42271829/10960229
def generate_param_grids(steps, param_grids):

    final_params = []

    for estimator_names in itertools.product(*steps.values()):
        current_grid = {}

        # Step_name and estimator_name should correspond
        # i.e preprocessor must be from pca and select.
        for step_name, estimator_name in zip(steps.keys(), estimator_names):
            for param, value in param_grids.get(estimator_name).items():
                if param == 'pipe_step_instance':
                    # Set actual estimator in pipeline
                    current_grid[step_name] = [value]
                else:
                    # Set parameters corresponding to above estimator
                    current_grid[step_name + '__' + param] = value
        # Append this dictionary to final params
        final_params.append(current_grid)

    return final_params

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