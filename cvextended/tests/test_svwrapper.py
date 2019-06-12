from ..svwrapper import SamplingWrapperSV
import pytest

from smote_variants import SMOTE
from imblearn.over_sampling import SMOTE as SMOTE2

def get_imb_learn_SMOTE():
    return SMOTE2()

def get_sv_SMOTE():
    return SMOTE()

def get_smote_like():
    pass

def test__validate_module():
    a = get_imb_learn_SMOTE()
    b = get_sv_SMOTE()

    with pytest.raises(TypeError):
        SamplingWrapperSV._validate_module(a)

    assert SamplingWrapperSV._validate_module(b)

def test__validate_methods():
    a = get_imb_learn_SMOTE()
    b = get_sv_SMOTE()

    with pytest.raises(AttributeError):
        SamplingWrapperSV._validate_methods(a)

    assert SamplingWrapperSV._validate_methods(b)