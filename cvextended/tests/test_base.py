from ..base import generate_param_grids
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid

def test_generate_param_grids():
    pipeline_steps = {'preprocessor':['skip'],
                  'classifier':['svm', 'rf']}
    all_params_grid = {
        'skip': {
            'pipe_step_instance': None
        },
        'svm': {
            'pipe_step_instance': SVC(probability=True),
            'C': [1, 10, 100],
            'gamma': [.01, .1],
            'kernel': ['rbf']
        },
        'rf': {
            'pipe_step_instance': RandomForestClassifier(),
            'n_estimators': [1, 10, 100],
            'max_features': [1, 5, 10, 20]
        }
    }
    out = generate_param_grids(steps=pipeline_steps, param_grids=all_params_grid)
    try:
        _ = ParameterGrid(out)
    except Exception as e:
        raise e
    print(out)
    assert 1!=1